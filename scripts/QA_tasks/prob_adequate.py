from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import json

device = "cuda" # the device to load the model onto

model_name = "facebook/opt-30b"  # 'facebook/opt-125m' 
dataset_name = 'ambg_coqa' #'ambig_qa' # 'trivia_qa' 'ambg_coqa'

path = '/home/eilia/semantic_entropy/output/sequences/ambig_qa/metrics/'
json_name = 'sequencesopt-30b_10_samples_with_prompt_0-1070_including_sem_ent_entail_knowledge_fact_check_llm_1_step_large_classifier_gpt-3.5-turbo_greedy_classifier_rougeL'

with open(path + json_name + '.json') as f:
    sequences = json.load(f)

#Data needed
if dataset_name == 'ambg_coqa':
    story = [item["few_shot_question"].split('Question:')[0].replace("Context: ","") for item in sequences]
    ambig_question = [item["question"] for item in sequences]
    generations = [item["cleaned_text"] for item in sequences]
    greedy = [sequences[i]["most_likely_generation"].replace(sequences[i]["few_shot_question"],"") for i in range(len(sequences))]
elif dataset_name == 'ambig_qa':
    ambig_question = [item["question"].replace('Question: ','').replace(' Answer: ','') for item in sequences]
    generations = [item["cleaned_text"] for item in sequences]
    greedy = [sequences[i]["most_likely_generation"].replace(sequences[i]["few_shot_question"],"") for i in range(len(sequences))]
else:
    raise TypeError('Enter valid dataset')

def construct_prompt(ambig_question, story, generations, possible_answer):
    #Construct p adequate prompt
    if story != "":
        prompt = f"Context: '{story}' \n Question: '{ambig_question}' \n Here are some brainstormed ideas: \n"
    else:
        prompt = f"Question: '{ambig_question}' \n Here are some brainstormed ideas: \n"
    
    for gen in generations:
        prompt += f"{gen} \n"
    
    prompt += f"Possible answer: {possible_answer} \n Is the possible answer: \n (A) Plausible \n (B) Not Plausible \n The possible answer is:"
   
    return prompt


def get_log_prob_from_loss(prompt, model, tokenizer):
    """Receives a sentence and produces its probability under the model"""
    inputs =  tokenizer(prompt, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    log_prob = - loss*( len(inputs["input_ids"][0]) - 1) #multiplying by length since the total loss 
    #was the average of losses, 1/N Sigma(L_CE) and subtracting 1 to not account for the </s> bos_token
    return log_prob.item()

def conditional_log_prob_of_continuation_given_context(log_prob_context, log_prob_context_and_continuation):
    """Receives the joint log probabilities for a context and the context and its continuation and returns the conditional
    log probability of the continuation given the context under the model"""

    log_prob_conditional_on_context = log_prob_context_and_continuation - log_prob_context
        
    return(log_prob_conditional_on_context)

def get_p_true(model, tokenizer, input_data):
    """Get the probability of the model anwering A (True) for the given input."""

    log_prob_prompt = get_log_prob_from_loss(input_data, model, tokenizer)
    log_prob_prompt_and_A = get_log_prob_from_loss(input_data + ' (A)', model, tokenizer)
    log_prob_prompt_and_B = get_log_prob_from_loss(input_data + ' (B)', model, tokenizer)

    log_prob_A_given_prompt = conditional_log_prob_of_continuation_given_context(log_prob_prompt, log_prob_prompt_and_A)
    log_prob_B_given_prompt = conditional_log_prob_of_continuation_given_context(log_prob_prompt, log_prob_prompt_and_B)

    log_prob_A = log_prob_A_given_prompt - torch.logsumexp(torch.Tensor([log_prob_A_given_prompt, log_prob_B_given_prompt]), dim = 0)
        
    return log_prob_A.item()


model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False, torch_dtype = torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

for i in range(len(ambig_question)):
    if dataset_name == 'ambg_coqa':
        prompt = construct_prompt(ambig_question[i],story[i], generations[i], greedy[i])
    elif dataset_name == 'ambig_qa':
        prompt = construct_prompt(ambig_question[i], '', generations[i], greedy[i])

    log_prob = get_p_true(model, tokenizer, prompt)
    
    sequences[i]['log_prob_true'] = log_prob


with open(f'{path}{json_name}_prob_adeq.json', 'w') as f:
    json.dump(sequences, f)
