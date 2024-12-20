from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import time
import openai
import pandas as pd

device = "cuda" # the device to load the model onto

adequacy_proxy = 'check_for_adequacy_fact_knowledge_check_llm_1_step'
# OPTIONS
# 'llm_plausible_check'
size_classifier = 'medium' # 'small' 'medium' 'large' 'x-large'
openai_key = True

dataset = 'ambig_qa'

path = '/home/eilia/semantic_entropy/output/sequences/ambig_qa/metrics/'
json_name = 'sequencesopt-30b_10_samples_with_prompt_0-1070_including_sem_ent_entail_knowledge_fact_check_llm_1_step_large_classifier'

with open(path + json_name + '.json') as f:
    sequences = json.load(f)

if dataset == 'abg_coqa':
    story = [item["few_shot_question"].split('Question:')[0].replace("Context: ","") for item in sequences]
    ambig_question = [item["question"] for item in sequences]
    greedy_gen = [sequences[i]["most_likely_generation"].replace(sequences[i]["few_shot_question"],"") for i in range(len(sequences))]
    references = [item["rougeL"]["plausible_answers"] for item in sequences]
elif dataset == 'ambig_qa':
    ambig_question = [item["question"].replace('Question: ','').replace(' Answer: ','') for item in sequences]
    greedy_gen = [sequences[i]["most_likely_generation"].replace(sequences[i]["few_shot_question"],"") for i in range(len(sequences))]
    ambigqa_data = pd.read_parquet('/home/eilia/semantic_entropy/fix_SE/data_ambig/validation-00000-of-00001-2.parquet')
    references = []
    for item in sequences:
        id_ = item['id']
        annotations = ambigqa_data[ambigqa_data['id'] == str(id_)]['annotations'].item()['qaPairs']
        plausible_answers = [x.tolist() for item in annotations for x in item['answer']]
        references.append(set([x for item in plausible_answers for x in item]))
else:
    raise TypeError('Enter valid dataset')

def check_for_adequacy_fact_check_llm_1_step(passage, question, references, proposed_answer, model, tokenizer, openai_key):
    prompt = f"You are presented with a document, a question based on the document, some acceptable answers and a proposed answer. Generate True if the proposed answer is a plausible answer to the question given the document and False if not. By plausible, I mean that the answer might be conveying the same meaning as one of the acceptable answers (even if it contains more or less information, as long as they have the same meaning), or, even if not similar to one of the acceptable answers, the answer can still be supported by the document. \nDocument:'{passage}' \nQuestion:'{question}' \nAcceptable Answers: '{references}' \nProposed Answer: '{proposed_answer}'."
    # prompt = f"You are presented with a passage, and a continuation to the passage. If the continuation is a word, generate True if it is a plausible continuation to the passage and False otherwise. If instead of a word, the continuation is a punctuation mark, generate True if it is plausible (i.e. the passage remains cohesive and comprehensive when adding that punctuation) and False otherwise. Only generate True or False. Passage:'{context}' Continuation:'{word}'."
    
    if openai_key == True:
        response = model.chat.completions.create(messages = [{"role": "user", "content" : prompt}], model="gpt-3.5-turbo",temperature = 0, top_p = 1, max_tokens = 200)
        print(response.choices[0])
        prediction = response.choices[0].message.content
        time.sleep(0.5)
    else:
        messages = [{"role": "user", "content": prompt}]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        # encodeds = tokenizer(prompt, return_tensors="pt")
        model_inputs = encodeds.to(device)
        model.to(device)

        # generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=False)
        decoded = tokenizer.batch_decode(generated_ids)
        # decoded = tokenizer.decode(generated_ids[0])
        # return decoded.replace(prompt, "")
        prediction = decoded[0].replace(prompt, "")
    return prediction

def check_for_adequacy_fact_knowledge_check_llm_1_step(question, references, proposed_answer, model, tokenizer, openai_key):
    prompt = f"You are presented with a question, some acceptable answers and a proposed answer. Generate True if the proposed answer is a plausible answer to the question given your training data and False if not. By plausible, I mean that the answer might be conveying the same meaning as one of the acceptable answers (even if it contains more or less information, as long as they have the same meaning), or, even if not similar to one of the acceptable answers, the answer can still be supported by your training data. \nQuestion:'{question}' \nAcceptable Answers: '{references}' \nProposed Answer: '{proposed_answer}'."
    # prompt = f"You are presented with a passage, and a continuation to the passage. If the continuation is a word, generate True if it is a plausible continuation to the passage and False otherwise. If instead of a word, the continuation is a punctuation mark, generate True if it is plausible (i.e. the passage remains cohesive and comprehensive when adding that punctuation) and False otherwise. Only generate True or False. Passage:'{context}' Continuation:'{word}'."
    
    if openai_key == True:
        response = model.chat.completions.create(messages = [{"role": "user", "content" : prompt}], model="gpt-3.5-turbo",temperature = 0, top_p = 1, max_tokens = 200)
        print(response.choices[0])
        prediction = response.choices[0].message.content
        time.sleep(0.5)
    else:
        messages = [{"role": "user", "content": prompt}]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        # encodeds = tokenizer(prompt, return_tensors="pt")
        model_inputs = encodeds.to(device)
        model.to(device)

        # generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=False)
        decoded = tokenizer.batch_decode(generated_ids)
        # decoded = tokenizer.decode(generated_ids[0])
        # return decoded.replace(prompt, "")
        prediction = decoded[0].replace(prompt, "")
    return prediction

if openai_key == True: 
    model_check = openai.OpenAI(api_key = '<OPEN_AI_API_KEY>')
    tokenizer_check = None
    model_name = 'openai/gpt-3.5-turbo'
else:
    if size_classifier == 'x-large':
        model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1' #46B parameters
    elif size_classifier == 'large':
        model_name = "google/gemma-2-27b-it"
        # model_name = 'mistralai/Mistral-Small-Instruct-2409' #22B parameters
    elif size_classifier == 'medium':
        model_name = 'mistralai/Mistral-Nemo-Instruct-2407' #12B parameters
    else: #if not specified, use the smallest one
        model_name = 'mistralai/Mistral-7B-Instruct-v0.2' #7B parameters

    model_check = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.bfloat16, device_map="auto")
    tokenizer_check = AutoTokenizer.from_pretrained(model_name)

for i in range(len(ambig_question)):
    if adequacy_proxy == 'fact_check_llm_1_step_plausible': 
        print('greedy_gen', greedy_gen[i])
        predicted_adeq_greedy = check_for_adequacy_fact_check_llm_1_step(story[i], ambig_question[i], references[i], greedy_gen[i], model_check, tokenizer_check, openai_key)
        predicted_adeq_greedy = predicted_adeq_greedy.replace("<s> [INST]  [/INST]","").replace("<s>[INST][/INST]","").replace("</s>", "")
        print('predicted_adeq_greedy',predicted_adeq_greedy)
    if adequacy_proxy == 'check_for_adequacy_fact_knowledge_check_llm_1_step': 
        print('greedy_gen', greedy_gen[i])
        predicted_adeq_greedy = check_for_adequacy_fact_knowledge_check_llm_1_step(ambig_question[i], references[i], greedy_gen[i], model_check, tokenizer_check, openai_key)
        predicted_adeq_greedy = predicted_adeq_greedy.replace("<s> [INST]  [/INST]","").replace("<s>[INST][/INST]","").replace("</s>", "")
        print('predicted_adeq_greedy',predicted_adeq_greedy)
    else:
        raise TypeError('Enter valid method of evaluating response adequacy')

    classifier_name = model_name.split('/')[1]
    sequences[i][f'greedy_plausible_{classifier_name}_classifier'] = predicted_adeq_greedy

with open(f'{path}{json_name}_{classifier_name}_greedy_classifier.json', 'w') as f:
    json.dump(sequences, f)
