import json
import os
import re
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
from typing import List
import argparse
import random
import sys

parser = argparse.ArgumentParser(description="Run inference with LLM")
parser.add_argument('--model_path', type=str, required=True, help='Path to the LLM')
parser.add_argument('--cuda_device', type=str, default='0', help='GPU device ID to use (default: "0")')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the results')
parser.add_argument('--shots_number', type=int, default=1, help='Number of shots to use')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.eval()

dataset = ['arithmetic', 'abstract_narrative_understanding', 'MMLU-Pro']

def read_json_objects(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        obj = json.loads(content)
    return obj

def Softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)
    return x

def create_user_prompt(question: str, options: List[str], shots):
        option_ids = list('ABCDEFGHIJK')
        options_str = "\n".join([f"{option_id}. {answer}".strip()
                        for option_id, answer in zip(option_ids, options)])
       
        user_prompt = f"""{shots}Question: {question.strip()}
Choices:
{options_str}
Answer: """  
        return user_prompt

def create_shot_prompt(question: str, options: List[str], answer: str):
        option_ids = list('ABCDEFGHIJK')
        options_str = "\n".join([f"{option_id}. {answer}".strip()
                        for option_id, answer in zip(option_ids, options)])
       
        user_prompt = f"""Question: {question.strip()}
Choices:
{options_str}
Answer:{answer}"""  
        return user_prompt

def create_zero_prompt(question: str, options: List[str]):
        option_ids = list('ABCDEFGHIJK')
        options_str = "\n".join([f"{option_id}. {answer}".strip()
                        for option_id, answer in zip(option_ids, options)])
       
        user_prompt = f"""Question: {question.strip()}
Choices:
{options_str}
Answer: """  
        return user_prompt

def make_shots(number, path):
    prompts = ''
    shot_data = read_json_objects(path)

    seed = 0
    random.seed(seed)

    shots = random.sample(shot_data, number)

    for shot in shots:
        question = shot['question']
        options = shot['choices']
        answer = shot['answer']
        prompt = create_shot_prompt(question, options, answer)
        prompts = prompts + prompt + '\n\n'

    return prompts

def get_token_pr(prompt, len):
    input_ids = tokenizer(prompt, return_tensors="pt")
    input_ids.to(device)
    with torch.no_grad():
        option_ids = list('ABCDEFGHIJK')[0:len]
        option_indices = [tokenizer(e).input_ids[-1] for e in option_ids]
    
        logits = model(
            **input_ids
        ).logits[:, -1].view(-1)
        
        prob = F.softmax(
            logits[option_indices], dim=-1
        ).detach().cpu().to(torch.float32).numpy()
       
        return prob
    
def cal_score(question, option):
    with torch.no_grad():
        q_tokens = tokenizer.encode(question, add_special_tokens=False)
        o_tokens = tokenizer.encode(option, add_special_tokens=False)

        input_ids = q_tokens + o_tokens[:-1]
        target_ids = o_tokens  

        input_tensor = torch.tensor([input_ids]).to(device)
        output = model(input_tensor)
        logits = output.logits[0, -len(target_ids):, :] 

        log  = F.log_softmax(logits, dim=-1)
        target_tensor = torch.tensor(target_ids).to(device)

        log_prob = log.gather(1, target_tensor.unsqueeze(-1)).squeeze(-1)
        total_log_prob = log_prob.sum().item()
         
        return total_log_prob, len(target_ids)

def create_permutation_options(options):
    default_options = options

    n = len(default_options)
    permutation = [list((i + j) % n for j in range(n)) for i in range(n)]

    shuffled_options = []
    for perm in permutation:
        shuffled_option = [default_options[idx] for idx in perm]
        shuffled_options.append(shuffled_option)

    return shuffled_options

def calculate_prob(all_prior, probs):
    length = len(probs)
    probs = probs / all_prior[:length]
    probs = probs / np.sum(probs)

    return probs

if __name__ == "__main__":
    model_name = model_path.split('/')[-1]

    for d in dataset:
        path = f'/dataset/pride/{d}_prior.json'
        data = read_json_objects(path)

        length = len(data[0]['choices'])
        all_prior = np.zeros(length, dtype=np.float32)
        for i in range(len(data)):
            case = data[i]
            question = case['question']
            option_id = list('ABCDEFGHIJK')[0:length]
            options = case['choices']

            options1 = create_permutation_options(options)
            prior = np.zeros(length, dtype=np.float32)
            for j, option in enumerate(options1):
                prompt = create_zero_prompt(question, option)
                prob = get_token_pr(prompt, len(option))
                prob = np.log(prob + 1e-10)
                prior += prob

            prior = prior / length
            prior = Softmax(prior)
            all_prior += prior

        all_prior = all_prior / len(data)



        path = f'/dataset/test/{d}.json'
        shot_path = f'/dataset/pride/{d}_prior.json'

        data = read_json_objects(path)
        shots = make_shots(args.shots_number, shot_path)
        for i in range(0, len(data)):
            case = data[i]
            question = case['question']
            option_id = list('ABCDEFGHIJK')[0:length]
            options = case['choices']

            
            # log
            prompt = create_user_prompt(question, options, shots)
            prob = get_token_pr(prompt, len(options))
            new_prob = calculate_prob(all_prior, prob)
            log_prob = np.log(new_prob + 1e-10)
            avg_log_prob = np.mean(log_prob)

            index = np.where(log_prob < avg_log_prob)[0]
            remain_index = np.where(log_prob >= avg_log_prob)[0]
            if len(index) == len(options) - 1:
                answer = option_id[np.argmax(log_prob)]
            else:
                new_options = []
                for j in range(len(options)):
                    if j not in index:
                        new_options.append(options[j])
                new_prompt = create_user_prompt(question, new_options, shots)
                new_prob = get_token_pr(new_prompt, len(new_options))
                new_prob = calculate_prob(all_prior, new_prob)

                index = remain_index[np.argmax(new_prob)]
                answer = option_id[index]

            path = args.output_path + f'/{args.shots_number}/{d}/{model_name}/log.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')

            
            # D-MCP
            prob = calculate_prob(all_prior, prob)
            index = np.argmax(prob)
            answer = option_id[index]

            path = args.output_path + f'/{args.shots_number}/{d}/{model_name}/D-MCP.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')