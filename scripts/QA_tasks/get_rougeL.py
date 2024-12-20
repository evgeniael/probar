import os
import random
import json

import numpy as np
import torch

import evaluate
import numpy as np
import torch
from collections import Counter
import pandas as pd

dataset = 'ambig_qa' # 'trivia_qa' 'ambig_qa' 'ambg_coqa'
correctness_threshold = 0.3

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

path = '/home/eilia/semantic_entropy/output/sequences/ambig_qa/metrics/' # path name to folder with .json file
json_name = 'sequencesopt-30b_10_samples_with_prompt_0-1070_including_sem_ent_entail_knowledge_fact_check_llm_1_step_large_classifier_gpt-3.5-turbo_greedy_classifier' # name of json file

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
    ambigqa_data = pd.read_parquet('raw_datasets/data_ambig_qa/validation-00000-of-00001-2.parquet')
    references = []
    for item in sequences:
        id_ = item['id']
        annotations = ambigqa_data[ambigqa_data['id'] == str(id_)]['annotations'].item()['qaPairs']
        plausible_answers = [x.tolist() for item in annotations for x in item['answer']]
        references.append(list(set([x for item in plausible_answers for x in item])))
else:
    raise TypeError('Enter valid dataset')

def get_estimator(sequences):
    seq_counter = dict(Counter(sequences))
    dist_support = seq_counter.keys()
    counts = list(seq_counter.values())
    dist_probs = [count/sum(counts) for count in counts]

    return dist_support, torch.tensor(dist_probs)

def entropy(probs):
    log_probs = torch.log(probs)
    entropy = - torch.sum(torch.mul(probs, log_probs))

    return entropy

def evaluate_rougeL(generation, reference, rouge_type = 'rougeL'):
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=[generation], references=[reference])
    score = rouge_results[rouge_type]#.mid.fmeasure
    return score


for count, sample in enumerate(sequences):
    id_ = sample['id']
    print(id_)
    generations = sample['cleaned_text']

    dist_support, dist_probs = get_estimator(generations)

    entropy_value = entropy(dist_probs)

    #Get entry of dataset with matching id
    if dataset == 'trivia_qa':
        id_column = 'question_id'
    else:
        id_column = 'id'
    
    plausible_answers = references[count]
    #Correctness of greedy generation is assessed as RougeL > correctness_threshold against any of the plausible answers 
    try:
        rougeL = [evaluate_rougeL(sample["most_likely_generation"].replace(sample["few_shot_question"], "").lstrip(), answer) for answer in plausible_answers]
    except:
        rougeL = [-1] #in case of NaN
        plausible_answers = 'NaN'

    correctness_to_plausible_answers = []
    for item in rougeL:
        if item > correctness_threshold:
            correctness_to_plausible_answers.append(1)
        else:
            correctness_to_plausible_answers.append(0)

    rougeL_dict = {'plausible_answers': plausible_answers, 'rougeL':rougeL}
    greedy_correct = sum(correctness_to_plausible_answers) > 0

    sample['generations_dist_support'] = dist_support
    sample['generations_dist_probs'] = dist_probs
    sample['entropy'] = entropy_value.item()
    sample['rougeL'] = rougeL_dict
    sample['greedy_correct'] = greedy_correct


for sequence in sequences:
    sequence['generations_dist_probs'] = sequence['generations_dist_probs'].tolist()
    sequence['generations_dist_support'] = list(sequence['generations_dist_support'])

with open(f'{path}{json_name}_rougeL.json', 'w') as fout:
    json.dump(sequences, fout)
