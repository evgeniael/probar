import os
import pickle
import random
import json

import numpy as np
import torch
import csv

import pathlib
from lib2to3.pgen2.tokenize import tokenize

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

device = 'cuda'
# Set a seed value
seed_value = 10

def define_seed(seed_value):
    # Set a seed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

define_seed(seed_value)

current_dir = os.getcwd()

size_classifier = 'medium'
method_equivalence = 'entailment' #'entailment' 'sem_equivalence'

def check_for_entailment_llm_1_step(s1, s2, model, tokenizer):
    prompt = f"You are presented with two strings, String 1 and String 2. Generate True if String 1 semantically entails String 2 and False otherwise. Only generate True or False. String 1:'{s1}' String 2:'{s2}'."
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=50, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0].replace(prompt, "")

def check_for_semantic_equivalence_llm_1_step(s1, s2, model, tokenizer):
    prompt = f"You are presented with two strings, String 1 and String 2. Generate True if String 1 and String 2 are semantically equivalent (i.e. have the same meaning) and False otherwise. Only generate True or False. String 1:'{s1}' String 2:'{s2}'."
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=50, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0].replace(prompt, "")


path = '/home/eilia/semantic_entropy/output/sequences/eilia.8348411/'
json_name = 'sequencesopt-30b_10_random_context_and_corrupted_samples_without_prompt_0-100'

with open(path + json_name + '.json') as f:
    sequences = json.load(f)

def process_generations(gen):
    gen = gen.split(' ')
    gen = [s for s in gen if s] #remove empty strings
    return gen[0]

ids = [item["id"] for item in sequences]
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

if size_classifier == 'x-large':
    model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1' #46B parameters
elif size_classifier == 'large':
    model_name = 'mistralai/Mistral-Small-Instruct-2409' #22B parameters
elif size_classifier == 'medium':
    model_name = 'mistralai/Mistral-Nemo-Instruct-2407' #12B parameters
else: #if not specified, use the smallest one
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2' #7B parameters

model_sem_equiv = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False, torch_dtype = torch.bfloat16, device_map="auto")
tokenizer_sem_equiv = AutoTokenizer.from_pretrained(model_name)

#This is the algorithm used for semantic clustering
run_id = 'run_1'

classifier_predictions = []
result_dict = {}

for k in range(len(context)):
    id_ = ids[k]

    generated_continuations = gen_words[k]
    unique_generated_texts = list(set(generated_continuations))

    response_list_1 = []
    response_list_2 = []
    inputs = []

    semantic_set_ids = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index

    if len(unique_generated_texts) > 1:
        for i, reference_answer in enumerate(unique_generated_texts): # Evalauate semantic similarity
            for j in range(i + 1, len(unique_generated_texts)):

                response_list_1.append(unique_generated_texts[i])
                response_list_2.append(unique_generated_texts[j])

                cw_1 = context[k] + ' ' + unique_generated_texts[i]
                cw_2 = context[k] + ' ' + unique_generated_texts[j]

                if method_equivalence == 'entailment':
                    prediction1 = check_for_entailment_llm_1_step(cw_1, cw_2, model_sem_equiv, tokenizer_sem_equiv)
                    prediction2 = check_for_entailment_llm_1_step(cw_2, cw_1, model_sem_equiv, tokenizer_sem_equiv)
                    condition_equiv = ('true' in prediction1.lower()) and ('true' in prediction2.lower())
                elif method_equivalence == 'sem_equivalence':
                    prediction = check_for_semantic_equivalence_llm_1_step(cw_1, cw_2, model_sem_equiv, tokenizer_sem_equiv)
                    condition_equiv = ('true' in prediction.lower())
                else:
                    raise TypeError('Set one of allowed methods to assess semantic equivalence')
                
                classifier_prediction = 0
                if condition_equiv: #semantic_equivalence
                    classifier_prediction = 1
                    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]     
                classifier_predictions.append([unique_generated_texts[i], unique_generated_texts[j], classifier_prediction])            
    
    result_dict[id_] = {}
    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_continuations]
    print('list_of_semantic_set_ids', list_of_semantic_set_ids)
    result_dict[id_]['semantic_set_ids'] = list_of_semantic_set_ids

pathlib.Path(f'{os.getcwd()}/clustering/').mkdir(parents=True, exist_ok=True)

with open(f'{run_id}_{json_name}.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(['cw_1', 'cw_2', 'prediction'])
    writer.writerows(classifier_predictions)

with open(f'{json_name}_generations_similarities.pkl', 'wb') as outfile:
    pickle.dump(result_dict, outfile)