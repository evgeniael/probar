import os
import pickle
import random
import json

import numpy as np
import torch

from rouge_score import rouge_scorer
import evaluate
import numpy as np
import torch
from collections import Counter

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

#Here, input the result from get_generations.py
path_name = '/home/eilia/semantic_entropy/output/sequences/ambig_qa/' # path to folder
file_name = 'sequencesopt-30b_10_samples_with_prompt_700-1070' #name of .json file, without .json

#Here, input the result from the equivalent get_clusters_entail.py, for the same dataset/subset
with open(path_name + file_name + '.json') as f:
    sequences = json.load(f)
with open('/home/eilia/semantic_entropy/output/sequences/ambig_qa/clustering/sequencesopt-30b_10_samples_with_prompt_700-1070_generations_similarities_entail.pkl', 'rb') as infile: 
    similarities_dict = pickle.load(infile)

def get_semantic_set_data(similarities_dict, sequences):
    #Adding the semantic set information to our sequence data
    with torch.no_grad():
        for sample in sequences:
            id_ = sample['id']
            sample['semantic_set_ids_entail'] = torch.tensor(similarities_dict[id_]['semantic_set_ids_entail'])#, device=device)
    
    return sequences

def get_estimator(sequences):
    #MC estimate distribution, given set of sequences
    seq_counter = dict(Counter(sequences))
    dist_support = seq_counter.keys()
    counts = list(seq_counter.values())
    dist_probs = [count/sum(counts) for count in counts]

    return dist_support, torch.tensor(dist_probs)

def entropy(probs):
    #Entropy approximation given torch.Tensor of probs
    log_probs = torch.log(probs)
    entropy = - torch.sum(torch.mul(probs, log_probs))

    return entropy

def align_semantics_sets_with_dist(sentences, semantic_sets, dist_support):
    # We create the support of the 'semantic' distribution
    dist_semantic_sets = []
    for sentence in dist_support:
        index = sentences.index(sentence)
        dist_semantic_sets.append(semantic_sets[index].item())
    
    return dist_semantic_sets

def semantic_entropy(probs, semantic_sets):
    # We get the probs of the 'semantic' distribution, and compute entropy over them
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

for sample in sequences:
    id_ = sample['id']
    print(id_)
    generations = sample['cleaned_text']

    dist_support, dist_probs = get_estimator(generations)

    dist_semantic_sets = align_semantics_sets_with_dist(generations, sample['semantic_set_ids_entail'], dist_support)
    semantic_entropy_value = semantic_entropy(dist_probs, dist_semantic_sets)

    sample['semantic_entropy_entail'] = semantic_entropy_value.item()


for sequence in sequences:
    sequence['semantic_set_ids_entail'] = sequence['semantic_set_ids_entail'].tolist()

with open(f'{path_name}metrics/{file_name}_including_sem_ent_entail.json', 'w') as fout:
    json.dump(sequences, fout)
