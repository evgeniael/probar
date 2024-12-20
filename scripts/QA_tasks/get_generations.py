import os
import pickle
import random
import json
import torch

import evaluate
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import Dataset
import pandas as pd

device = 'cuda'
current_dir = os.getcwd()

def define_seed(seed_value):
    # Set a seed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

seed_value = 10
define_seed(seed_value)

num_generations_per_prompt = 10
model_name = "facebook/opt-30b"  # 'facebook/opt-2.7b' 'facebook/opt-6.7b' 'facebook/opt-13b' 'facebook/opt-30b'
model_naming = 'opt-30b' # "opt-2.7b" 'opt-6.7b' 'opt-13b' 'opt-30b'
instruction_tuned = False

dataset_name = 'ambig_qa' #'ambig_qa' 'ambg_coqa' 'provo_corpus'

if dataset_name == 'provo_corpus':
    sampling_method = 'corrupt' #or 'random'

if dataset_name == 'ambg_coqa':
    dataset_part = 'test' #'train' 'dev' 'test' -- relevant to abgcoqa
    type_instances = "non_ambiguous" #"non_ambiguous" "ambiguous"-- relevant to abgcoqa

decoding_method = 'greedy'
temperature = 1.0
top_p = 1.0
#few-shot prompt
prompt = 'yes' # 'yes' 'no'
hugging_face_token = False #True if token is needed to access model

#Indices of the dataset for which we want to obtain generations
sample_index_lower = 0
sample_index_upper = 1070

batch_number = 5 #the batch size of inputs when generating



if (hugging_face_token == True):
    api_token = '<API_TOKEN>'

model = AutoModelForCausalLM.from_pretrained(f"{model_name}", use_cache=False, 
                                            torch_dtype = torch.bfloat16, 
                                            device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(f"{model_name}", use_fast=False) #cache_dir=config.data_dir)
tokenizer.padding_side='left'

if instruction_tuned == True:
    tokenizer.pad_token = tokenizer.eos_token

if dataset_name == 'ambig_qa':

    if not os.path.exists(os.path.join(os.getcwd(), '/ambig_qa')):

        print('Preprocessing dataset')
        val_data = pd.read_parquet('/raw_datasets/data_ambig_qa/validation-00000-of-00001-2.parquet')
        train_data = pd.read_parquet('/raw_datasets/data_ambig_qa/train-00000-of-00001-2.parquet')

        batch_size = 1

        def process_data_to_model_inputs(batch, few_shot_prompt):
            # tokenize the inputs and labels
            dict_results = {}
            # we are only interested in the set of questions that are ambiguous for the purposes of our experiments
            answers = []
            disambiguated_questions = []
            batch_ambiguous_questions = []
            id = []
            annotations = []

            for i in range(len(batch['annotations'])):
                if batch['annotations'].values[i]['type'][0] == "multipleQAs":
                    list_of_answers = batch['annotations'].values[i]['qaPairs'][0]['answer']
                    list_of_answers = [item[0] for item in list_of_answers]
                    answers.append("<sep_answer>".join(list_of_answers))
                    disambiguated_questions.append(batch['annotations'].values[i]['qaPairs'][0]['question'])
                    batch_ambiguous_questions.append(batch['question'][i])
                    id.append(batch['id'][i])
                    annotations.append(batch['annotations'][i])
            
            if prompt == 'yes':
                batch_with_prompt = [few_shot_prompt + " Question: " + question + " Answer: " for question in batch_ambiguous_questions] #batch['question']
            else:
                batch_with_prompt = [" Question: " + question + " Answer: " for question in batch_ambiguous_questions] #batch['question']
            
            if instruction_tuned == True:
                batch_with_prompt = ['<s>[INST] ' + prompt + ' [/INST]' for prompt in batch_with_prompt]

            inputs = tokenizer(batch_with_prompt, padding=True, truncation=False)
            outputs = tokenizer(answers, padding=True, truncation=False)

            dict_results["question"] = batch_ambiguous_questions
            dict_results["id"] = id
            dict_results["annotations"] = annotations
            dict_results["input_ids"] = inputs.input_ids
            dict_results["attention_mask"] = inputs.attention_mask
            dict_results["decoder_input_ids"] = outputs.input_ids
            dict_results["decoder_attention_mask"] = outputs.attention_mask
            dict_results["labels"] = outputs.input_ids.copy()
            dict_results['answer'] = answers
            dict_results['disambiguated_questions'] = disambiguated_questions
            
            # print('batch_labels',batch['labels'])
            # print('amb_labels',amb_labels)
            # # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
            # # We have to make sure that the PAD token is ignored
            dict_results["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in dict_results["labels"]]

            return dict_results

        if prompt == 'yes': #use training data to get few shot prompt
            len_few_shot = 10
            data_for_few_shot_prompt = train_data[:len_few_shot]

            examples =''
            for i in range(len(data_for_few_shot_prompt['annotations'])):
                if data_for_few_shot_prompt['annotations'].values[i]['type'][0] == "singleAnswer":
                    list_of_answers = data_for_few_shot_prompt['annotations'].values[i]['answer'][0].tolist()
                    examples = examples + ' Question: ' + data_for_few_shot_prompt['question'][i] + ' Answer: ' + list_of_answers[0]
                else:
                    list_of_answers = data_for_few_shot_prompt['annotations'].values[i]['qaPairs'][0]['answer'][0].tolist()
                    examples = examples + ' Question: ' + data_for_few_shot_prompt['question'][i] + ' Answer: ' + list_of_answers[0]
                # if data_for_few_shot_prompt['annotations'].values[i]['type'][0] == "multipleQAs":
                #     list_of_answers = data_for_few_shot_prompt['annotations'].values[i]['qaPairs'][0]['answer'][0].tolist()
                #     examples = examples + ' Question: ' + data_for_few_shot_prompt['question'][i] + ' Answer: ' + list_of_answers[0]
            
            few_shot_prompt = examples

            val_data = process_data_to_model_inputs(val_data.drop(columns=["viewed_doc_titles", "used_queries", "nq_doc_title"]), few_shot_prompt = few_shot_prompt)
        else:
            val_data = process_data_to_model_inputs(val_data.drop(columns=["viewed_doc_titles", "used_queries", "nq_doc_title"]), few_shot_prompt = '')

        pd.DataFrame.from_dict(val_data).to_csv('ambig_qa.csv', index=False)  
    else:

        val_data = pd.read_csv('ambig_qa.csv')
elif dataset_name == 'ambg_coqa':
    if not os.path.exists(os.path.join(os.getcwd(), '/ambg_coqa')):

        print('Preprocessing dataset')
        with open('/raw_datasets/data_ambg_coqa/coqa_abg_test.json', 'r') as j:
            test_data = json.loads(j.read())

        with open('/raw_datasets/data_ambg_coqa/coqa_abg_val.json', 'r') as j:
            val_data = json.loads(j.read())

        with open('/raw_datasets/data_ambg_coqa/coqa_abg_train.json', 'r') as j:
            train_data = json.loads(j.read())

        batch_size = 1

        def process_data_to_model_inputs(data_dict_batch, type_instances, prompt):
            batch = {}
            # tokenize the inputs and labels
            answers = []
            ambg_question = []
            previous_turns = []
            story = []
            chat_completion_prompts = []
            # chat_completion_prompts_without_ambig_question = []
            dialogue_histories = []
            ids = []
            for item in data_dict_batch:
                if item["ambiguity"] == type_instances:
                    story.append(item["story"])
                    ids.append(item["id"])
                    previous_turns.append(item["history_turns"])
                    ambg_question.append(item["target_turn"]["question"])

                    if type_instances == "ambiguous":
                        plausible_answers = [x["org_ans"] for x in item["clarification_turn"]["answers"]]
                        plausible_answers.append(item["clarification_turn"]["question"])
                        #valid answers might either be a clarification question or one of the plausible answers
                        answers.append("<sep_answer>".join(plausible_answers))
                    else:
                        answers.append(item["target_turn"]["answer"])
                    
                    if prompt == "no":
                        raise TypeError("Prompting is mandatory for this dataset (Format: Context - Previous Questions - Previous Answers")
                    else:
                        dialogue_history = [item["story"]]
                        prompt = "Context: " + item["story"]
                        for turn in item["history_turns"]:
                            dialogue_history.append('Question: ' + turn["question"])
                            prompt = prompt + '\n Question:' + turn["question"]
                            dialogue_history.append('Answer: ' + turn["answer"] + '.') #newly added
                            prompt = prompt + '\n Answer:' + turn["answer"] + '.'
                        dialogue_history.append(item["target_turn"]["question"])

                        chat_completion_prompt = prompt + '\n Question:' + item["target_turn"]["question"] + '\n Answer:'
                        chat_completion_prompts.append(chat_completion_prompt)
                        dialogue_histories.append(dialogue_history)
            
            inputs = tokenizer(chat_completion_prompts, padding=True, truncation=False)
            outputs = tokenizer(answers, padding=True, truncation=False)

            batch["id"] = ids
            batch["answer"] = answers
            batch["ambg_question"] = ambg_question
            batch["previous_turns"] = previous_turns
            batch["story"] = story
            batch["chat_completion_prompt"] = chat_completion_prompts
            batch["dialogue_history"] = dialogue_histories
            batch["input_ids"] = inputs.input_ids
            batch["attention_mask"] = inputs.attention_mask
            batch["decoder_input_ids"] = outputs.input_ids
            batch["decoder_attention_mask"] = outputs.attention_mask
            batch["labels"] = outputs.input_ids.copy()

            # # because BERT automatically shifts the labels, the labels correspond exactly to decoder_input_ids.
            # # We have to make sure that the PAD token is ignored
            batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

            return batch
        
        if prompt == 'no':
            raise TypeError('Prompting is mandatory for this dataset (Format: Context - Previous Questions - Previous Answers')
        else:
            if dataset_part == 'test':
                data = process_data_to_model_inputs(test_data["data"], type_instances, prompt)
            elif dataset_part == 'dev':
                data = process_data_to_model_inputs(val_data["data"], type_instances, prompt)
            else:
                data = process_data_to_model_inputs(train_data["data"], type_instances, prompt)

        data = pd.DataFrame.from_dict(data)
        data.to_csv(f"ambg_coqa_{dataset_part}_{type_instances}.csv", index=False)
    else:
        data = pd.read_csv(f'ambg_coqa_{dataset_part}_{type_instances}.csv')
elif dataset_name == 'provo_corpus':
    if not os.path.exists(os.path.join(os.getcwd(), '/provo_corpus')):

        print('Preprocessing dataset')
        test_data = pd.read_csv('/raw_datasets/data_provo/Provo_Corpus.tsv', sep='\t')
        
        batch_size = 1

        def process_data_to_model_inputs(predict_norms, prompt):
            batch = {}
            # tokenize the inputs and labels
            answers = []
            context = []
            ids = []

            paragraphs = predict_norms.groupby('Text_ID')['Text'].max()
            for text_id in range(1,len(paragraphs)+1): #iterate over all provo paragraphs
                for word_num in predict_norms[predict_norms['Text_ID'] == text_id]['Word_Number'].unique(): #iterating over all words in each text
                    word_dist = predict_norms[(predict_norms['Text_ID'] == text_id) & (predict_norms['Word_Number'] == word_num)]
                    unique_human_words = list(word_dist['Response'].unique()) #all human answered words for each word
                    unique_human_words = [x for x in unique_human_words if str(x) != 'nan']
                    gold_label = paragraphs[text_id].split(' ')[int(word_num)-1:int(word_num)]
                    if not (gold_label[0] in unique_human_words):
                        unique_human_words.append(gold_label[0])
                    answers.append("<sep_answer>".join(unique_human_words))
                    context.append(" ".join(paragraphs[text_id].split(' ')[:int(word_num)-1]))
                    ids.append(word_dist['Word_Unique_ID'].unique()[0])
                    #store the distribution dictionary in a dictionary for each text word (and their position in the text - in case of duplicate words)                       
            
            inputs = tokenizer(context, padding=True, truncation=False)
            outputs = tokenizer(answers, padding=True, truncation=False)

            batch["question_id"] = ids
            batch["answer"] = answers
            batch["prompt"] = context
            batch["input_ids"] = inputs.input_ids
            batch["attention_mask"] = inputs.attention_mask
            batch["decoder_input_ids"] = outputs.input_ids
            batch["decoder_attention_mask"] = outputs.attention_mask
            batch["labels"] = outputs.input_ids.copy()

            # # because BERT automatically shifts the labels, the labels correspond exactly to decoder_input_ids.
            # # We have to make sure that the PAD token is ignored
            batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

            return batch
        
        if prompt == 'yes':
            raise TypeError('No prompting for next-word prediction task')
        else:
            data = process_data_to_model_inputs(test_data, prompt)

        data = pd.DataFrame.from_dict(data)
        data.to_csv(f"provo_corpus.csv", index=False)
    else:
        data = pd.read_csv(f'provo_corpus.csv')
else:
    raise TypeError("Use an existing dataset")


if dataset_name == 'ambig_qa':
    dataset = pd.read_csv(f'{os.getcwd()}/ambig_qa.csv')
elif dataset_name == 'ambg_coqa':
    dataset = pd.read_csv(f'{os.getcwd()}/ambg_coqa_{dataset_part}_{type_instances}.csv')
elif dataset_name == 'provo_corpus':
    dataset = pd.read_csv(f'{os.getcwd()}/provo_corpus.csv')


class AmbigQADataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.input_ids = df["input_ids"] 
        self.attention_mask = df["attention_mask"] 
        self.decoder_input_ids = df["decoder_input_ids"] 
        self.decoder_attention_mask = df["decoder_attention_mask"]
        self.labels = df["labels"] 
        self.answer = df['answer'] 
        self.question = df['question'] 
        self.id = df['id']
        self.disambig_questions = df['disambiguated_questions']

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input_ids':torch.LongTensor(json.loads(self.input_ids[idx])), 
                'attention_mask': torch.LongTensor(json.loads(self.attention_mask[idx])),
                'decoder_input_ids': torch.LongTensor(json.loads(self.decoder_input_ids[idx])), 
                'decoder_attention_mask': torch.LongTensor(json.loads(self.decoder_attention_mask[idx])), 
                'labels':torch.LongTensor(json.loads(self.labels[idx])), 
                'answer': self.answer[idx], 
                'question': self.question[idx],
                'question_id': self.id[idx],
                'disambiguated_questions': self.disambig_questions[idx]}
        return sample

class AmbgCoqaDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.input_ids = df["input_ids"] 
        self.attention_mask = df["attention_mask"] 
        self.decoder_input_ids = df["decoder_input_ids"] 
        self.decoder_attention_mask = df["decoder_attention_mask"]
        self.labels = df["labels"] 
        self.answer = df['answer'] 
        self.question = df['ambg_question']
        self.id = df['id']       
        self.previous_turns = df['previous_turns']
        self.story = df['story']
        self.chat_completion_prompt = df['chat_completion_prompt']
        self.dialogue_history = df['dialogue_history']

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input_ids':torch.LongTensor(json.loads(self.input_ids[idx])), 
                'attention_mask': torch.LongTensor(json.loads(self.attention_mask[idx])),
                'decoder_input_ids': torch.LongTensor(json.loads(self.decoder_input_ids[idx])), 
                'decoder_attention_mask': torch.LongTensor(json.loads(self.decoder_attention_mask[idx])), 
                'labels':torch.LongTensor(json.loads(self.labels[idx])), 
                'answer': self.answer[idx], 
                'question': self.question[idx],
                'question_id': self.id[idx],
                'previous_turns': self.previous_turns[idx],
                'story': self.story[idx],
                'chat_completion_prompt': self.chat_completion_prompt[idx],
                'dialogue_history':self.dialogue_history[idx]}
        return sample

class ProvoCorpusDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.input_ids = df["input_ids"] 
        self.attention_mask = df["attention_mask"] 
        self.decoder_input_ids = df["decoder_input_ids"] 
        self.decoder_attention_mask = df["decoder_attention_mask"]
        self.labels = df["labels"] 
        self.answer = df['answer'] 
        self.id = df['id']       
        self.prompt = df['prompt']

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input_ids':torch.LongTensor(json.loads(self.input_ids[idx])), 
                'attention_mask': torch.LongTensor(json.loads(self.attention_mask[idx])),
                'decoder_input_ids': torch.LongTensor(json.loads(self.decoder_input_ids[idx])), 
                'decoder_attention_mask': torch.LongTensor(json.loads(self.decoder_attention_mask[idx])), 
                'labels':torch.LongTensor(json.loads(self.labels[idx])), 
                'answer': self.answer[idx], 
                'id': self.id[idx],
                'prompt': self.prompt[idx]}
        return sample

if dataset_name == 'ambig_qa':
    questions = AmbigQADataset(dataset.iloc[sample_index_lower:sample_index_upper,:].set_axis(list(range(sample_index_upper - sample_index_lower)), axis='index'))
    dataloader = torch.utils.data.DataLoader(questions, batch_size=batch_number)
elif dataset_name == 'ambg_coqa':
    if type_instances == 'ambiguous':
        questions = AmbgCoqaDataset(dataset.iloc[sample_index_lower:sample_index_upper,:].set_axis(list(range(sample_index_upper - sample_index_lower)), axis='index'))
        dataloader = torch.utils.data.DataLoader(questions, batch_size=batch_number)
    else:
        len_dataset = dataset.shape[0]
        num_samples = sample_index_upper - sample_index_lower
        sampled_indeces = random.sample(range(len_dataset), num_samples)
        questions = AmbgCoqaDataset(dataset.iloc[sampled_indeces,:].set_axis(list(range(num_samples)), axis='index'))
        dataloader = torch.utils.data.DataLoader(questions, batch_size=batch_number)
elif dataset_name == 'provo_corpus':
    if sampling_method == 'corrupt':
        num_random_context = sample_index_upper - sample_index_lower
        corrupt_prompts_path = '/processed_datasets/provo_corpus/provo_corpus_corrupted_prompts.json'
        with open(corrupt_prompts_path, 'r') as file:
            corrupt_prompts = json.load(file)
        corrupt_ids = []
        for key in corrupt_prompts.keys():
            corrupt_ids.append(corrupt_prompts[key][0])
        assert (len(corrupt_ids) == num_random_context)
        filtered_corrupt = dataset[dataset['id'].isin(corrupt_ids)]
        contexts = ProvoCorpusDataset(filtered_corrupt.set_axis(list(range(filtered_corrupt.shape[0])), axis='index'))
        dataloader = torch.utils.data.DataLoader(contexts, batch_size=batch_number)
    else:
        num_random_context = sample_index_upper - sample_index_lower
        contexts = ProvoCorpusDataset(dataset.sample(n=num_random_context, replace=False).set_axis(list(range(num_random_context)), axis='index'))
        dataloader = torch.utils.data.DataLoader(contexts, batch_size=batch_number)
else:
    raise TypeError('Enter valid dataset')

period_token_id = tokenizer('. ')['input_ids'][1]
eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]
eos = [item for row in question_framing_ids for item in row]
rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")

def get_generations(model, dataloader, number_of_generations, batch_number, max_tokens=50, compute_rouge = 'No'):
    """For a given model, produce a number of generation """

    with torch.no_grad():
        sequences = []
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            if decoding_method == 'greedy':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=1,
                                                        do_sample=False,
                                                        max_new_tokens = max_tokens,
                                                        eos_token_id=eos,
                                                        use_cache = False)
            
            
            generations = model.generate(input_ids, do_sample=True, num_return_sequences=number_of_generations,
                                    num_beams=1,
                                    max_new_tokens = max_tokens, 
                                    eos_token_id=eos,
                                    temperature=temperature,
                                    top_p=top_p,
                                    use_cache = False)
            original_len = generations.shape[1]
            batch_size = input_ids.shape[0]
            generations = generations.unsqueeze_(0).view(batch_size,number_of_generations,original_len)

            batch_size = input_ids.shape[0]

            for i in range(batch_size):
                few_shot_question = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                if prompt == 'yes':
                    if dataset_name == 'ambg_coqa':
                        question = batch['question'][i]
                    else:
                        question = few_shot_question.replace(few_shot_prompt, "")
                else:
                    question = few_shot_question
                
                sequence_dict = {
                        'prompt': input_ids[i],
                        'generations': generations[i],
                        'id': batch['question_id'][i],
                        'few_shot_question': few_shot_question,
                        'question': question,
                        'generated_text': tokenizer.batch_decode(generations[i,:,:], skip_special_tokens=True),
                        'most_likely_generation_ids': most_likely_generation[i][0].to('cpu'),
                        'most_likely_generation': tokenizer.decode(most_likely_generation[i], skip_special_tokens=True)
                    }
                
                if dataset_name == 'ambig_qa':
                    sequence_dict['disambiguated_questions'] = batch['disambiguated_questions'][i]
                elif dataset_name == 'ambg_coqa':
                    sequence_dict['ambg_question'] = batch['ambg_question'][i]
                    sequence_dict['previous_turns'] = batch['previous_turns'][i]
                    sequence_dict['story'] = batch['story'][i]
                    sequence_dict['chat_completion_prompts'] = batch['chat_completion_prompts'][i]
                elif dataset_name == 'provo_corpus':
                    sequence_dict['answer'] = batch['answer'][i]

                cleaned_text = [item.replace(few_shot_question, "") for item in sequence_dict['generated_text']]
                
                if instruction_tuned == True:
                    cleaned_text = [item.replace(" [/INST]", "") for item in cleaned_text]
                    cleaned_text = [item.replace(".", "") for item in cleaned_text]

                sequence_dict['cleaned_text'] = cleaned_text

                if compute_rouge == 'yes':
                    rouge_types = ['rouge1', 'rouge2', 'rougeL']
                    for rouge_type in rouge_types:
                        if rouge_type in batch:
                            sequence_dict[rouge_type + '_reference_answers'] = batch[rouge_type]

                        else:
                            sequence_dict[rouge_type + '_reference_answers'] = None

                        sequence_dict[rouge_type + '_to_target'] = 0.0

                    sequence_dict['answer'] = batch['answer'][i]

                    sequence_dict['exact_match'] = 0.0

                    reference_answers = batch['answer'][i].split("<sep_answer>")
                    print(reference_answers)

                    for answer in reference_answers:
                        predictions = [sequence_dict['most_likely_generation'].lstrip()]
                        references = [answer]
                        results = exact_match_metric.compute(predictions=predictions,
                                                            references=references,
                                                            ignore_case=True,
                                                            ignore_punctuation=True)
                        sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
                        rouge_results = rouge.compute(predictions=predictions, references=references)
                        for rouge_type in rouge_types:
                            sequence_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type], sequence_dict[rouge_type + '_to_target'])

                sequences.append(sequence_dict)
    return sequences

sequences = get_generations(model, dataloader, num_generations_per_prompt, batch_number, max_tokens = 5) #for next word prediction


if prompt == 'yes':
    with open(f'/home/eilia/semantic_entropy/output/sequences/ambig_qa/{model_naming}_{num_generations_per_prompt}_samples_generations_with_prompt_{sample_index_lower}-{sample_index_upper}.pkl', 'wb') as outfile:
        pickle.dump(sequences, outfile)
else:
    with open(f'{model_naming}_{num_generations_per_prompt}_samples_generations_without_prompt_{sample_index_lower}-{sample_index_upper}.pkl', 'wb') as outfile:
        pickle.dump(sequences, outfile)

for sequence in sequences:
    sequence['prompt'] = sequence['prompt'].tolist()
    sequence['generations'] = sequence['generations'].tolist()
    sequence['most_likely_generation_ids'] = sequence['most_likely_generation_ids'].tolist()

if dataset_name == 'ambig_qa':
    for sequence in sequences:
        sequence['id'] = sequence['id'].tolist()

if prompt == 'yes':
    with open(f'/home/eilia/semantic_entropy/output/sequences/ambig_qa/sequences{model_naming}_{num_generations_per_prompt}_samples_with_prompt_{sample_index_lower}-{sample_index_upper}.json', 'w') as fout:
        json.dump(sequences, fout)
else:
    with open(f'/home/eilia/semantic_entropy/output/sequences/ambig_qa/sequences{model_naming}_{num_generations_per_prompt}_samples_without_prompt_{sample_index_lower}-{sample_index_upper}.json', 'w') as fout:
        json.dump(sequences, fout)