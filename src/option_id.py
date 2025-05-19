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

parser = argparse.ArgumentParser(description="Run inference with LLM")
parser.add_argument('--model_path', type=str, required=True, help='Path to the LLM')
parser.add_argument('--cuda_device', type=str, default='0', help='GPU device ID to use (default: "0")')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the results')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.eval()

dataset = ['arithmetic', 'timedial', 'abstract_narrative_understanding', 'MMLU-Pro']

def read_json_objects(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        obj = json.loads(content)
    return obj

def Softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)
    return x

def create_user_prompt(question: str, options: List[str]):
        option_ids = list('ABCDEFGHIJK')
        options_str = "\n".join([f"{option_id}. {answer}".strip()
                        for option_id, answer in zip(option_ids, options)])
       
        user_prompt = f"""Question: {question.strip()}
Choices:
{options_str}
Answer: """  
        return user_prompt

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
    
def create_permutation_options(options):
    default_options = options

    n = len(default_options)
    permutation = [list((i + j) % n for j in range(n)) for i in range(n)]

    shuffled_options = []
    for perm in permutation:
        shuffled_option = [default_options[idx] for idx in perm]
        shuffled_options.append(shuffled_option)

    return shuffled_options

def get_top_indices(prob):
    origin_indices = np.arange(len(prob))
    min_value = np.min(prob)
    min_indices = np.where(prob == min_value)[0].tolist()
    if len(min_indices) > 1:
        return origin_indices
    indices = np.delete(origin_indices, min_indices)
    return indices

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
                prompt = create_user_prompt(question, option)
                prob = get_token_pr(prompt, len(option))
                prob = np.log(prob + 1e-10)
                prior += prob

            prior = prior / length
            prior = Softmax(prior)
            all_prior += prior

        all_prior = all_prior / len(data)
       

        path = f'/dataset/test/{d}.json'
        data = read_json_objects(path)
        for i in range(0, len(data)):
            case = data[i]
            question = case['question']
            option_id = list('ABCDEFGHIJK')[0:length]
            options = case['choices']

            
            # log
            prompt = create_user_prompt(question, options)
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
                user_prompt = create_user_prompt(question, new_options)
                new_prob = get_token_pr(user_prompt, len(new_options))
                new_prob = calculate_prob(all_prior, new_prob)
                
                index = remain_index[np.argmax(new_prob)]
                answer = option_id[index]

            path = args.output_path + f'/{d}/{model_name}/log.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')



            # MCP
            index = np.argmax(prob)
            answer = option_id[index]

            path = args.output_path + f'/{d}/{model_name}/MCP.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')
          


            # D-MCP
            prob = calculate_prob(all_prior, prob)
            index = np.argmax(prob)
            answer = option_id[index]

            path = args.output_path + f'/{d}/{model_name}/D-MCP.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')


            
            # seq
            prompt = create_user_prompt(question, options)
            prob = get_token_pr(prompt, len(options))
            prob = calculate_prob(all_prior, prob)
            new_options = options.copy()

            for j in range(0, length-1):
                index = get_top_indices(prob)
                index.sort()
                new_options  = [new_options[k] for k in index]

                prompt = create_user_prompt(question, new_options)
                prob = get_token_pr(prompt, len(new_options))
                prob = calculate_prob(all_prior, prob)

            answer_index = np.argmax(prob)
            answer_index = options.index(new_options[answer_index])

            answer = option_id[answer_index]

            path = args.output_path + f'/{d}/{model_name}/seq.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')

