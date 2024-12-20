from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import openai
import time

device = "cuda" # the device to load the model onto

adequacy_proxy = 'fact_check_llm_1_step_plausible'
# OPTIONS
# 'llm_plausible_check'
size_classifier = 'medium' # 'small' 'medium' 'large' 'x-large'
openai_key = False

dataset = 'provo_corpus'

path = '/home/eilia/semantic_entropy/output/metrics/next_word_pred/random_context_including_metrics/eilia.8441792/'
json_name = 'sequencesopt-30b_10_random_context_and_corrupted_samples_without_prompt_0-100_fact_check_llm_1_step_plausible_medium_classifier_including_metrics'

with open(path + json_name + '.json') as f:
    sequences = json.load(f)

def process_generations(gen):
    gen = gen.split(' ')
    gen = [s for s in gen if s] #remove empty strings
    return gen[0]

context = [item["question"] for item in sequences]
generations = [item["cleaned_text"] for item in sequences]
greedy_gen = [sequences[i]["most_likely_generation"].replace(context[i],"") for i in range(len(sequences))]
greedy_gen_word = [process_generations(w) for w in greedy_gen]

gen_words = []
for gens in generations:
    l_gen = []
    for g in gens:
        l_gen.append(process_generations(g))
    gen_words.append(l_gen)

def check_for_adequacy_fact_check_llm_1_step(context, word,  model, tokenizer, openai_key):
    prompt = f"You are presented with a piece of text and a continuation. Generate True if the continuation is a plausible continuation to the context and False if it is not a plausible continuation. By plausible, I mean that when concatenating the continuation to the text, the text will remain grammatically correct and comprehensible. Text:'{context}' Continuation:'{word}' Answer:"
    # prompt = f"You are presented with a passage, and a continuation to the passage. If the continuation is a word, generate True if it is a plausible continuation to the passage and False otherwise. If instead of a word, the continuation is a punctuation mark, generate True if it is plausible (i.e. the passage remains cohesive and comprehensive when adding that punctuation) and False otherwise. Only generate True or False. Passage:'{context}' Continuation:'{word}'."

    if openai_key == True:
        response = model.chat.completions.create(messages = [{"role": "user", "content" : prompt}], model="gpt-3.5-turbo",temperature = 0, top_p = 1, max_tokens = 200)
        print(response.choices[0])
        prediction = response.choices[0].message.content
        time.sleep(0.5)
    else:
        # messages = [{"role": "user", "content": prompt}]
        # encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        encodeds = tokenizer(prompt, return_tensors="pt")
        model_inputs = encodeds.to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        # decoded = tokenizer.batch_decode(generated_ids)
        decoded = tokenizer.decode(generated_ids[0])
        prediction = decoded.replace(prompt, "")
    
    return prediction
    # return decoded[0].replace(prompt, "")

if openai_key == True:
    # openai.organization = "insert_organization"
    # openai.api_key = 
    model_check = openai.OpenAI(api_key = '<OPEN_AI_API_KEY>')
    tokenizer_check = None
else:
    if size_classifier == 'x-large':
        model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1' #46B parameters
    elif size_classifier == 'large':
        # model_name = "google/gemma-2-27b-it"
        model_name = 'mistralai/Mistral-Small-Instruct-2409' #22B parameters
    elif size_classifier == 'medium':
        model_name = 'mistralai/Mistral-Nemo-Instruct-2407' #12B parameters
    else: #if not specified, use the smallest one
        model_name = 'mistralai/Mistral-7B-Instruct-v0.2' #7B parameters

    model_check = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.bfloat16, device_map="auto")
    tokenizer_check = AutoTokenizer.from_pretrained(model_name)

for i in range(len(context)):
    if adequacy_proxy == 'fact_check_llm_1_step_plausible': 
        #in the case that we will only have one model
        unique_gens = set(gen_words[i]) #this might contain repeating items, so we iterate only over unique responses
        print('unique_gens',unique_gens)
        predicted_adeq = [check_for_adequacy_fact_check_llm_1_step(context[i], gen, model_check, tokenizer_check, openai_key) for gen in unique_gens]
        predicted_adeq = [x.replace("<s> [INST]  [/INST]","").replace("<s>[INST][/INST]","").replace("</s>", "") for x in predicted_adeq]
        declarative_sentences = 'Not relevant to 1-step LLM fact checking'
        print('predicted_adeq',predicted_adeq)
        #map unique gens and their adequacy preds to initial gens
        dict_adeq = dict(zip(unique_gens, predicted_adeq))
        predicted_support = [dict_adeq[gen] for gen in gen_words[i]]
        predicted_adeq_greedy = check_for_adequacy_fact_check_llm_1_step(context[i], greedy_gen_word[i], model_check, tokenizer_check, openai_key)
        predicted_adeq_greedy = predicted_adeq_greedy.replace("<s> [INST]  [/INST]","").replace("<s>[INST][/INST]","").replace("</s>", "")
        print('predicted_adeq_greedy',predicted_adeq_greedy)
    else:
        raise TypeError('Enter valid method of mapping claims/QA pairs to adequacy')

    sequences[i][f'declarative_sentences_mistral_{size_classifier}_classifier'] = declarative_sentences
    sequences[i][f'predicted_plausible_mistral_{size_classifier}_classifier'] = predicted_support
    sequences[i][f'greedy_plausible_mistral_{size_classifier}_classifier'] = predicted_adeq_greedy

with open(f'{json_name}_{adequacy_proxy}_{size_classifier}_classifier.json', 'w') as f:
    json.dump(sequences, f)
