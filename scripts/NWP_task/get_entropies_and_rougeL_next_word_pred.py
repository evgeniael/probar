import os
import pickle
import random
import json

import numpy as np
import torch

import pathlib
from rouge_score import rouge_scorer
import evaluate
import numpy as np
import torch
from collections import Counter
import pandas as pd

# Set a seed value
seed_value = 10
correctness_threshold = 0.3

def define_seed(seed_value):
    # Set a seed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

define_seed(seed_value)

current_dir = os.getcwd()

path = '/home/eilia/semantic_entropy/output/metrics/eilia.8440437/'
json_name = 'sequencesopt-6.7b_10_random_context_and_corrupted_samples_without_prompt_0-100_fact_check_llm_1_step_plausible_medium_classifier'

with open(path + json_name + '.json') as f:
    sequences = json.load(f)

with open('/home/eilia/semantic_entropy/output/clustering/eilia.8440237/sequencesopt-6.7b_10_random_context_and_corrupted_samples_without_prompt_0-100_generations_similarities.pkl', 'rb') as infile:
    similarities_dict = pickle.load(infile)

def get_semantic_set_data(similarities_dict, sequences):
    #Adding the semantic set information to our sequence data
    with torch.no_grad():
        for sample in sequences:
            id_ = sample['id']
            sample['semantic_set_ids'] = torch.tensor(similarities_dict[id_]['semantic_set_ids'])#, device=device)
    
    return sequences

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

def align_semantics_sets_with_dist(sentences, semantic_sets, dist_support):
    dist_semantic_sets = []
    for sentence in dist_support:
        index = sentences.index(sentence)
        dist_semantic_sets.append(semantic_sets[index].item())
    
    return dist_semantic_sets

def semantic_entropy(probs, semantic_sets):
    classes = set(semantic_sets)
    probs_classes = []
    for c in classes:
        probs_items_in_class = torch.where((torch.LongTensor(semantic_sets) == c), probs, torch.zeros(len(probs)))
        prob_class = torch.sum(probs_items_in_class)
        probs_classes.append(prob_class)
    
    sem_entropy = entropy(torch.tensor(probs_classes))
    # return classes, probs_classes, sem_entropy
    return sem_entropy

sequences = get_semantic_set_data(similarities_dict, sequences)

def evaluate_rougeL(generation, reference, rouge_type = 'rougeL'):
    # scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # score = scorer.score(generation,reference)
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=[generation], references=[reference])
    score = rouge_results[rouge_type]#.mid.fmeasure
    return score

def process_generations(gen):
    gen = gen.split(' ')
    gen = [s for s in gen if s] #remove empty strings
    return gen[0]

context = [item["question"] for item in sequences]
generations = [item["cleaned_text"] for item in sequences]
greedy_gen = [sequences[i]["most_likely_generation"].replace(context[i],"") for i in range(len(sequences))]
greedy_gen_word = [process_generations(w) for w in greedy_gen]

for sample in sequences:
    id_ = sample['id']
    generations = sample['cleaned_text']
    gen_words = []
    for g in generations:
        gen_words.append(process_generations(g))
    
    dist_support, dist_probs = get_estimator(gen_words)

    entropy_value = entropy(dist_probs)
    dist_semantic_sets = align_semantics_sets_with_dist(gen_words, sample['semantic_set_ids'], dist_support)
    semantic_entropy_value = semantic_entropy(dist_probs, dist_semantic_sets)

    plausible_answers = sample['answer'].split('<sep_answer>')

    #Correctness of greedy generation is assessed as RougeL > correctness_threshold against any of the plausible answers 
    try:
        rougeL = [evaluate_rougeL(process_generations(sample["most_likely_generation"].replace(sample["question"], "").lstrip()), answer) for answer in plausible_answers]
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
    greedy_correct_rougeL = sum(correctness_to_plausible_answers) > 0

    sample['generations_dist_support'] = dist_support
    sample['generations_dist_probs'] = dist_probs
    sample['entropy'] = entropy_value.item()
    sample['semantic_entropy'] = semantic_entropy_value.item()
    sample['rougeL'] = rougeL_dict
    sample['greedy_correct_rougeL'] = greedy_correct_rougeL


pathlib.Path(f'{os.getcwd()}/metrics/').mkdir(parents=True, exist_ok=True)

for sequence in sequences:
    sequence['generations_dist_probs'] = sequence['generations_dist_probs'].tolist()
    sequence['semantic_set_ids'] = sequence['semantic_set_ids'].tolist()
    sequence['generations_dist_support'] = list(sequence['generations_dist_support'])

with open(f'{json_name}_including_metrics.json', 'w') as fout:
    json.dump(sequences, fout)
