from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import json

device = "cuda" # the device to load the model onto

adequacy_proxy = 'knowledge_fact_check_llm_1_step'
# OPTIONS
# 'NLI_llm_2_steps', 'NLI_hard', 'NLI_easy'
# 'fact_check_llm_1_step' 'knowledge_fact_check_llm_1_step' 'fact_check_llm_1_step_chain_of_thought' 'fact_check_llm_1_step_plausible'
# 'fact_check_llm_2_steps_wording_support' 'fact_check_llm_2_steps_wording_support_few_shot' 'fact_check_llm_2_steps_wording_not_contra'
size_classifier = 'large' # 'small' 'medium' 'large' 'x-large'

dataset = 'ambig_qa'

path = '/home/eilia/semantic_entropy/output/sequences/ambig_qa/metrics/'
json_name = 'sequencesopt-30b_10_samples_with_prompt_0-1070_including_sem_ent_entail'

with open(path + json_name + '.json') as f:
    sequences = json.load(f)

#Data needed
if dataset == 'abg_coqa':
    story = [item["few_shot_question"].split('Question:')[0].replace("Context: ","") for item in sequences]
    ambig_question = [item["question"] for item in sequences]
    generations = [item["cleaned_text"] for item in sequences]
elif dataset == 'ambig_qa':
    ambig_question = [item["question"].replace('Question: ','').replace(' Answer: ','') for item in sequences]
    generations = [item["cleaned_text"] for item in sequences]
else:
    raise TypeError('Enter valid dataset')

def get_declarative_sentence(ambig_question, generation):
    #assertive vs declarative vs affirmative 
    prompt = f"Turn a question-answer pair to a declarative sentence. Only output the sentence and nothing else. Question: '{ambig_question}' Answer: '{generation}'"
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer_qa_declaration.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model_qa_declaration.to(device)

    generated_ids = model_qa_declaration.generate(model_inputs, max_new_tokens=100, do_sample=False)
    decoded = tokenizer_qa_declaration.batch_decode(generated_ids)

    return decoded[0].replace(prompt, "")

def check_for_adequacy_nli(story, declarative_sentence):
    # max_length = 256
    premise = story
    hypothesis = declarative_sentence

    tokenized_input_seq_pair = tokenizer_nli.encode_plus(premise, hypothesis, #  max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)

    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model_nli(input_ids.to(device="cuda"),
                    attention_mask=attention_mask.to(device="cuda"),
                    token_type_ids=token_type_ids.to(device="cuda"),
                    labels=None)
    
    if adequacy_proxy == 'NLI_hard':
        id2label = {"0": "entailment", "1": "neutral", "2": "contradiction"} #this is the mapping for the labels for nli_hard model
    else:
        id2label =  {"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"} #this is for nli_easy model

    predicted_probs = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    return (predicted_probs, id2label)

def check_for_adequacy_fact_check_llm_1_step(story, question, answer):
    prompt = f"You are presented with a document, a question based on the document and an answer to the question. Generate True if the answer to the question is supported by the document and False if the answer to the question is not supported by the document. Only generate True or False. Document: '{story}' Question:'{question}' Answer:'{answer}'."
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer_fact_check.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model_fact_check.to(device)

    generated_ids = model_fact_check.generate(model_inputs, max_new_tokens=50, do_sample=False)
    decoded = tokenizer_fact_check.batch_decode(generated_ids)

    return decoded[0].replace(prompt, "")

def check_for_adequacy_fact_check_llm_1_step_plausible(story, question, answer):
    prompt = f"You are presented with a document, a question based on the document and an answer to the question. Generate True if the answer to the question is plausible given the document and False if the answer to the question is not plausible given the document. Only generate True or False. Document: '{story}' Question:'{question}' Answer:'{answer}'."
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer_fact_check.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model_fact_check.to(device)

    generated_ids = model_fact_check.generate(model_inputs, max_new_tokens=50, do_sample=False)
    decoded = tokenizer_fact_check.batch_decode(generated_ids)

    return decoded[0].replace(prompt, "")

def check_for_adequacy_fact_check_llm_1_step_chain_of_thought(story, question, answer):
    prompt = f"You are presented with a document, a question based on the document and an answer to the question. Generate True if the answer to the question is plausible given the document and False if the answer to the question is not plausible given the document. Generate your intermediate reasoning steps before generating your final answer. Document: '{story}' Question:'{question}' Answer:'{answer}' Reasoning:"
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer_fact_check.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model_fact_check.to(device)

    generated_ids = model_fact_check.generate(model_inputs, max_new_tokens=200, do_sample=False)
    decoded = tokenizer_fact_check.batch_decode(generated_ids)

    return decoded[0].replace(prompt, "")

def check_for_adequacy_fact_check_llm_1_step_chain_of_thought_few_shot(story, question, answer):
    prompt = f"You are presented with a document, a question based on the document and an answer to the question. Generate True if the answer to the question is plausible given the document and False if the answer to the question is not plausible given the document. To reach an answer, generate your intermediate reasoning steps and at the end generate True or False. Document: '{story}' Question:'{question}' Answer:'{answer}' Reasoning:"
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer_fact_check.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model_fact_check.to(device)

    generated_ids = model_fact_check.generate(model_inputs, max_new_tokens=150, do_sample=False)
    decoded = tokenizer_fact_check.batch_decode(generated_ids)

    return decoded[0].replace(prompt, "")

def check_for_adequacy_fact_check_llm(story, declarative_sentence, few_shot_example = ''):
    if (adequacy_proxy == 'fact_check_llm_2_steps_wording_support'):
        prompt = f"You are presented with a document and a claim. Generate True if the claim is supported by the document and False if the claim is not supported by the document. Only generate True or False. Document: '{story}' Claim:'{declarative_sentence}'"
    elif (adequacy_proxy == 'fact_check_llm_2_steps_wording_not_contra'):
        prompt = f"You are presented with a document and a claim. Generate True if the claim is not contradicted by the document and False if the claim is contradicted by the document. Only generate True or False. Document: '{story}' Claim:'{declarative_sentence}'"
    else: #few shot prompt
        prompt = f"You are presented with a document and a claim. Generate True if the claim is supported by the document and False if the claim is not supported by the document. Only generate True or False, as in the following example: {few_shot_example} \n Document: '{story}' Claim:'{declarative_sentence} Prediction:'"
        
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer_fact_check.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model_fact_check.to(device)

    generated_ids = model_fact_check.generate(model_inputs, max_new_tokens=50, do_sample=False)
    decoded = tokenizer_fact_check.batch_decode(generated_ids)
    print(decoded[0])

    return decoded[0].replace(prompt, "")

def check_for_adequacy_knowledge_fact_check_llm_1_step(question, answer):
    prompt = f"You are presented with a question and an answer. Generate True if the answer is a plausible response to the question with respect to your training data and False if not. Only generate True or False. Question:'{question}' Answer:'{answer}'."
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer_fact_check.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model_fact_check.to(device)

    generated_ids = model_fact_check.generate(model_inputs, max_new_tokens=50, do_sample=False)
    decoded = tokenizer_fact_check.batch_decode(generated_ids)

    return decoded[0].replace(prompt, "")

def check_for_adequacy_nli_llm(premise, hypothesis, binary = True):
    if binary == True: #classifying true/entail vs false/not entail
        prompt = f"You are given a premise and a hypothesis. Generate True if the hypothesis is entailed by the premise and False if the hypothesis is not entailed by the premise. Only generate True or False. Premise: '{premise}' Hypothesis:'{hypothesis}'"
    else: #classifying entail, contradict or neutral
        prompt = f"You are given a premise and a hypothesis. Generate Entailment if the hypothesis is entailed by the premise, Contradiction if the hypothesis is contradicted by the premise and Neutral if not possible to determine. Only generate Entailment, Contradiction or Neutral. Premise: '{premise}' Hypothesis:'{hypothesis}'"

    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer_nli.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model_nli.to(device)

    generated_ids = model_nli.generate(model_inputs, max_new_tokens=50, do_sample=False)
    decoded = tokenizer_nli.batch_decode(generated_ids)
    print(decoded[0])

    return decoded[0].replace(prompt, "")


if (adequacy_proxy == 'fact_check_llm_2_steps_wording_support') or (adequacy_proxy == 'fact_check_llm_2_steps_wording_support_few_shot') or (adequacy_proxy == 'fact_check_llm_2_steps_wording_not_contra'):
    model_qa_declaration = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_cache=False, 
                                            torch_dtype = torch.bfloat16, 
                                            device_map="auto")
    tokenizer_qa_declaration = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    model_fact_check = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_cache=False, 
                                                torch_dtype = torch.bfloat16, 
                                                device_map="auto")
    tokenizer_fact_check = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
elif (adequacy_proxy == 'fact_check_llm_1_step') or (adequacy_proxy == 'fact_check_llm_1_step_plausible') or (adequacy_proxy == 'fact_check_llm_1_step_chain_of_thought') or (adequacy_proxy == 'knowledge_fact_check_llm_1_step'):
    if size_classifier == 'x-large':
        model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1' #46B parameters
    elif size_classifier == 'large':
        model_name = 'mistralai/Mistral-Small-Instruct-2409' #22B parameters
    elif size_classifier == 'medium':
        model_name = 'mistralai/Mistral-Nemo-Instruct-2407' #12B parameters
    else: #if not specified, use the smallest one
        model_name = 'mistralai/Mistral-7B-Instruct-v0.2' #7B parameters

    model_fact_check = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False, 
                                                torch_dtype = torch.bfloat16, 
                                                device_map="auto")
    tokenizer_fact_check = AutoTokenizer.from_pretrained(model_name)
elif adequacy_proxy == 'NLI_llm_2_steps':
    if size_classifier == 'x-large':
        model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1' #46B parameters
    elif size_classifier == 'large':
        model_name = 'mistralai/Mistral-Small-Instruct-2409' #22B parameters
    elif size_classifier == 'medium':
        model_name = 'mistralai/Mistral-Nemo-Instruct-2407' #12B parameters
    else: #if not specified, use the smallest one
        model_name = 'mistralai/Mistral-7B-Instruct-v0.2' #7B parameters

    model_qa_declaration = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', use_cache=False, 
                                            torch_dtype = torch.bfloat16, 
                                            device_map="auto")
    tokenizer_qa_declaration = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')

    model_nli = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False, 
                                                torch_dtype = torch.bfloat16, 
                                                device_map="auto")
    tokenizer_nli = AutoTokenizer.from_pretrained(model_name)
elif adequacy_proxy == 'NLI_hard':
    model_qa_declaration = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_cache=False, 
                                            torch_dtype = torch.bfloat16, 
                                            device_map="auto")
    tokenizer_qa_declaration = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    tokenizer_nli = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    model_nli = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli").cuda()
elif adequacy_proxy == 'NLI_easy':
    model_qa_declaration = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_cache=False, 
                                            torch_dtype = torch.bfloat16, 
                                            device_map="auto")
    tokenizer_qa_declaration = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    tokenizer_nli = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    model_nli = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda() 
else:
    raise TypeError('Provide valid algorithm for mapping to adequacy')

for i in range(len(ambig_question)):
    if adequacy_proxy == 'fact_check_llm_1_step': 
        #in the case that we will only have one model
        predicted_support = [check_for_adequacy_fact_check_llm_1_step(story[i], ambig_question[i], gen) for gen in generations[i]]
        predicted_support = [x.replace("<s> [INST]  [/INST]","").replace("<s>[INST][/INST]","").replace("</s>", "") for x in predicted_support]
        declarative_sentences = 'Not relevant to 1-step LLM fact checking'
    elif adequacy_proxy == 'fact_check_llm_1_step_plausible':
        predicted_support = [check_for_adequacy_fact_check_llm_1_step_plausible(story[i], ambig_question[i], gen) for gen in generations[i]]
        predicted_support = [x.replace("<s> [INST]  [/INST]","").replace("<s>[INST][/INST]","").replace("</s>", "") for x in predicted_support]
        declarative_sentences = 'Not relevant to 1-step LLM fact checking'
    elif adequacy_proxy == 'fact_check_llm_1_step_chain_of_thought':
        predicted_support = [check_for_adequacy_fact_check_llm_1_step_chain_of_thought(story[i], ambig_question[i], gen) for gen in generations[i]]
        predicted_support = [x.replace("<s> [INST]  [/INST]","").replace("<s>[INST][/INST]","").replace("</s>", "") for x in predicted_support]
        declarative_sentences = 'Not relevant to 1-step LLM fact checking Chain of Thought'
    elif adequacy_proxy == 'knowledge_fact_check_llm_1_step':
        predicted_support = [check_for_adequacy_knowledge_fact_check_llm_1_step(ambig_question[i], gen) for gen in generations[i]]
        predicted_support = [x.replace("<s> [INST]  [/INST]","").replace("</s>", "") for x in predicted_support]
        declarative_sentences = 'Not relevant to 1-step LLM fact checking'
    else: 
        #this is 2 steps methods, so initially we need to construct the claim from the QA pair
        declarative_sentences = [get_declarative_sentence(ambig_question[i], gen) for gen in generations[i]]
        declarative_sentences = [x.replace("<s> [INST]  [/INST]","").replace("</s>", "") for x in declarative_sentences]
        
        if (adequacy_proxy == 'NLI_hard') or (adequacy_proxy == 'NLI_easy'):
            predicted_support = [check_for_adequacy_nli(story[i], declarative_sent) for declarative_sent in declarative_sentences]
        elif adequacy_proxy == 'NLI_llm_2_steps':
            predicted_support = [check_for_adequacy_nli_llm(story[i], declarative_sent) for declarative_sent in declarative_sentences]
            predicted_support = [x.replace("<s> [INST]  [/INST]","").replace("<s>[INST][/INST]","").replace("</s>", "") for x in predicted_support]
        elif (adequacy_proxy == 'fact_check_llm_2_steps_wording_support') or (adequacy_proxy == 'fact_check_llm_2_steps_wording_not_contra'):
            predicted_support = [check_for_adequacy_fact_check_llm(story[i], declarative_sent) for declarative_sent in declarative_sentences]
            predicted_support = [x.replace("<s> [INST]  [/INST]","").replace("</s>", "") for x in predicted_support]
        elif adequacy_proxy == 'fact_check_llm_2_steps_wording_support_few_shot':
            #from validation set of ABG-COQA
            few_shot_example = "Document: 'The story of the day I lost my best friend to a car accident. The day a precious life was taken from us way too soon. \n\nIt was a bright and Sunny day in November. Thanksgiving had been celebrated only two days before. Since it was a holiday weekend I had been on the phone with Greg the night before many times. His dad didn't want him to come over because of the holiday. I guess he finally wore him down and he called and said, \"I can stay\". So, my mom, brother, and I went to pick him up. He was always smiling. The complete opposite of my shy self, Greg was always the life of the party. \n\nWe got two large pizzas that Friday night. I've never known anyone in my entire life who loved to eat more than Greg. That's the way he was though. He was just enjoying life. And if it meant gaining weight or whatever, so be it. He would sit back and put his hands on his belly and just laugh. We (Greg, David, and I) did so many funny things together and had such great times. Things we should have done and things we shouldn't have done, I'll \"Never\" forget. \n\nOn Saturday morning Dad took us out for breakfast. We all finished eating and followed my Dad up to the cashier. Greg asked Dad if he could have a candy bar. I looked at Greg shaking my head. He just laughed. After breakfast, Father took us to my Mom's house. \n\nWhen we got out at Mom's house there was no one home. So, one of us grabbed a big wheel and rode it down the steep driveway into the street. Just boys being boys. Greg and I did it several times until the last time. The car hit him on the head, knocking him around 75-- 100 yards. My brother and I both ran screaming just yelling for help and crying. One of the neighbors called 911. I was in shock. That day was forever etched into our memories. \n\nIt still hurts to think about it. Wishing we could have grown old together. Wondering how it would have been. I'm sure It WOULD HAVE BEEN GREAT.' Claim:'Everyone was running screaming and yelling for help and crying, and a neighbor called 911.' Prediction: True \n Document: 'The story of the day I lost my best friend to a car accident. The day a precious life was taken from us way too soon. \n\nIt was a bright and Sunny day in November. Thanksgiving had been celebrated only two days before. Since it was a holiday weekend I had been on the phone with Greg the night before many times. His dad didn't want him to come over because of the holiday. I guess he finally wore him down and he called and said, \"I can stay\". So, my mom, brother, and I went to pick him up. He was always smiling. The complete opposite of my shy self, Greg was always the life of the party. \n\nWe got two large pizzas that Friday night. I've never known anyone in my entire life who loved to eat more than Greg. That's the way he was though. He was just enjoying life. And if it meant gaining weight or whatever, so be it. He would sit back and put his hands on his belly and just laugh. We (Greg, David, and I) did so many funny things together and had such great times. Things we should have done and things we shouldn't have done, I'll \"Never\" forget. \n\nOn Saturday morning Dad took us out for breakfast. We all finished eating and followed my Dad up to the cashier. Greg asked Dad if he could have a candy bar. I looked at Greg shaking my head. He just laughed. After breakfast, Father took us to my Mom's house. \n\nWhen we got out at Mom's house there was no one home. So, one of us grabbed a big wheel and rode it down the steep driveway into the street. Just boys being boys. Greg and I did it several times until the last time. The car hit him on the head, knocking him around 75-- 100 yards. My brother and I both ran screaming just yelling for help and crying. One of the neighbors called 911. I was in shock. That day was forever etched into our memories. \n\nIt still hurts to think about it. Wishing we could have grown old together. Wondering how it would have been. I'm sure It WOULD HAVE BEEN GREAT.' Claim:'Nobody was yelling for help or crying.' Prediction: False"
            predicted_support = [check_for_adequacy_fact_check_llm(story[i], declarative_sent, few_shot_example) for declarative_sent in declarative_sentences]
            predicted_support = [x.replace("<s> [INST]  [/INST]","").replace("</s>", "") for x in predicted_support]
        else:
            raise TypeError('Enter valid method of mapping claims/QA pairs to adequacy')

    sequences[i][f'declarative_sentences_{size_classifier}_classifier'] = declarative_sentences
    sequences[i][f'predicted_support_{size_classifier}_classifier'] = predicted_support

with open(f'{path}{json_name}_{adequacy_proxy}_{size_classifier}_classifier.json', 'w') as f:
    json.dump(sequences, f)
