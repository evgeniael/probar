from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

device = "cuda" # the device to load the model onto

model_name = "facebook/opt-6.7b"  # 'facebook/opt-125m' 'google/gemma-2b-it'  'mistralai/Mistral-7B-Instruct-v0.2'
dataset_name = 'provo_corpus' 

path = '/home/eilia/semantic_entropy/output/metrics/next_word_pred/random_context_including_metrics/eilia.8486152/'
json_name = 'sequencesopt-6.7b_10_random_context_and_corrupted_samples_without_prompt_0-100_fact_check_llm_1_step_plausible_medium_classifier_including_metrics'

with open(path + json_name + '.json') as f:
    sequences = json.load(f)

def process_generations(gen):
    gen = gen.split(' ')
    gen = [s for s in gen if s] #remove empty strings
    return gen[0]

#Data needed
if dataset_name == 'provo_corpus':
    open_ended_context = [item["question"] for item in sequences]
    generations = [item["cleaned_text"] for item in sequences]
    greedy_gen = [sequences[i]["most_likely_generation"].replace(open_ended_context[i],"") for i in range(len(sequences))]
    greedy_gen_word = [process_generations(w) for w in greedy_gen]
else:
    raise TypeError('Enter valid dataset')

def construct_prompt(context, generations, possible_answer):
    #Construct p adequate prompt
    prompt = f"Context: '{context}' \n Here are some brainstormed continuations: \n"
    
    for gen in generations:
        prompt += f"{gen} \n"
    
    prompt += f"Possible continuation: {possible_answer} \n Is the possible continuation: \n (A) Plausible \n (B) Not Plausible \n The possible continuation is:"
   
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

for i in range(len(open_ended_context)):
    continuations = generations[i]
    gen_words = []
    for g in continuations:
        gen_words.append(process_generations(g))
    
    prompt = construct_prompt(open_ended_context[i], gen_words, greedy_gen_word[i])

    log_prob = get_p_true(model, tokenizer, prompt)
    
    sequences[i]['log_prob_true'] = log_prob


with open(f'{path}{json_name}_prob_adeq.json', 'w') as f:
    json.dump(sequences, f)
