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

dataset = ['arithmetic_A', 'arithmetic']

def read_json_objects(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        obj = json.loads(content)
    return obj

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

if __name__ == "__main__":
    model_name = model_path.split('/')[-1]

    for d in dataset:
        path = f'/dataset/test/{d}.json'
        data = read_json_objects(path)
        length = len(data[0]['choices'])
        for i in range(0, len(data)):
            case = data[i]
            question = case['question']
            option_id = list('ABCDEFGHIJK')[0:length]
            options = case['choices']

            
            # no_pride_log
            prompt = create_user_prompt(question, options)
            prob = get_token_pr(prompt, len(options))
            new_prob = prob.copy()
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
                
                index = remain_index[np.argmax(new_prob)]
                answer = option_id[index]

            path = args.output_path + f'/{d}/{model_name}/no_pride_log.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{answer}\n')