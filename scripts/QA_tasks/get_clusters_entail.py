import os
import pickle
import random
import json

import numpy as np
import torch
import csv

from lib2to3.pgen2.tokenize import tokenize

import evaluate
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

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

dataset_name = 'ambg_coqa' #'trivia_qa' 'ambg_coqa' 'ambig_qa'

path_name = '/home/eilia/semantic_entropy/output/sequences/ambig_qa/' #path to folder with file
file_name = 'sequencesopt-30b_10_samples_with_prompt_700-1070' #generations name of .json file

with open(path_name + file_name + '.json') as f:
    sequences = json.load(f)

for sequence in sequences:
  sequence['prompt'] = torch.LongTensor(sequence['prompt'])
  sequence['generations'] = torch.LongTensor(sequence['generations'])
  sequence['most_likely_generation_ids'] = torch.LongTensor(sequence['most_likely_generation_ids'])


#This is the NLI algorithm used for the clustering
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()

run_id = 'run_1'

deberta_predictions = []
result_dict = {}

#This script is based on Kuhn et al's implementation of the bi-directional entailment for semantic equivalence
# https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/get_semantic_similarities.py

for sample in sequences:
    question = sample['question'].replace("Question:","").replace("Answer:","")
    generated_texts = sample['cleaned_text']
    
    if dataset_name == 'trivia_qa':
        id_ = sample['id']
    elif dataset_name == 'ambig_qa':
        id_ = sample['id']
    elif dataset_name == 'ambg_coqa':
        id_ = sample['id']

    unique_generated_texts = list(set(generated_texts))

    answer_list_1 = []
    answer_list_2 = []
    has_semantically_different_answers = True
    inputs = []
    syntactic_similarities = {}

    semantic_set_ids = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index

    if len(unique_generated_texts) > 1:

        # Evalauate semantic similarity
        for i, reference_answer in enumerate(unique_generated_texts):
            for j in range(i + 1, len(unique_generated_texts)):

                answer_list_1.append(unique_generated_texts[i])
                answer_list_2.append(unique_generated_texts[j])

                qa_1 = question + ' ' + unique_generated_texts[i]
                qa_2 = question + ' ' + unique_generated_texts[j]

                input = qa_1 + ' [SEP] ' + qa_2

                inputs.append(input)
                encoded_input = tokenizer.encode(input, padding=True)
                prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                predicted_label = torch.argmax(prediction, dim=1)

                reverse_input = qa_2 + ' [SEP] ' + qa_1
                encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                deberta_prediction = 0  
                if (2 in predicted_label) and (2 in reverse_predicted_label): #if both ways are predicting entailment
                    has_semantically_different_answers = False
                    deberta_prediction = 1 
                    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]                    

                deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])
            
    result_dict[id_] = {'has_semantically_different_answers': has_semantically_different_answers}
        
    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]

    result_dict[id_]['semantic_set_ids_entail'] = list_of_semantic_set_ids


with open(f'{path_name}/clustering/{run_id}_{file_name}.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(['qa_1', 'qa_2', 'prediction'])
    writer.writerows(deberta_predictions)

with open(f'{path_name}/clustering/{file_name}_generations_similarities_entail.pkl', 'wb') as outfile:
    pickle.dump(result_dict, outfile)