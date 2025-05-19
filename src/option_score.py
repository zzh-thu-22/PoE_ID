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

def create_base_prompt(question: str):
        user_prompt = f"""Question: {question.strip()}
Answer: """ 
        return user_prompt

def create_prompt(question: str, options: List[str]):
        option_ids = list('ABCDEFGHIJK')
        options_str = "\n".join([f"{option_id}. {answer}".strip()
                        for option_id, answer in zip(option_ids, options)])
        
        user_prompt = f"""Question: {question.strip()}
Choices:
{options_str}
Answer: """ 
        return user_prompt

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

            LM_scores = np.zeros(len(options))
            AVG_scores = np.zeros(len(options))
            channel_scores = np.zeros(len(options))
            PMI_scores = np.zeros(len(options))
            AOLP_scores = np.zeros(len(options))
            for j, option in enumerate(options):
                domain_text = 'Answer:'

                prompt = create_base_prompt(question)
                AO_prompt = create_prompt(question, options)

                log_prob, num_token = cal_score(prompt, option)
                ch_log_prob, ch_num_token = cal_score(option, prompt)
                AO_log_prob, AO_num_token = cal_score(AO_prompt, option)
                do_log_prob, do_num_token = cal_score(domain_text, option)

                LM_scores[j] = np.exp(log_prob)
                AVG_scores[j] = log_prob / num_token
                channel_scores[j] = ch_log_prob / ch_num_token
                AOLP_scores[j] = AO_log_prob
                PMI_scores[j] = log_prob - do_log_prob
            
            LM_index = np.argmax(LM_scores)
            AVG_index = np.argmax(AVG_scores)
            channel_index = np.argmax(channel_scores)
            AOLP_index = np.argmax(AOLP_scores)
            PMI_index = np.argmax(PMI_scores)

            LM_answer = option_id[LM_index]
            AVG_answer = option_id[AVG_index]
            channel_answer = option_id[channel_index]
            AOLP_answer = option_id[AOLP_index]
            PMI_answer = option_id[PMI_index]

            path = args.output_path + f'/{d}/{model_name}/LM.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{LM_answer}\n')

            path = args.output_path + f'/{d}/{model_name}/AVG.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{AVG_answer}\n')

            path = args.output_path + f'/{d}/{model_name}/channel.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{channel_answer}\n')

            path = args.output_path + f'/{d}/{model_name}/AOLP.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{AOLP_answer}\n')

            path = args.output_path + f'/{d}/{model_name}/PMI.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{PMI_answer}\n')



            avg_score = np.mean(AOLP_scores)
            mask_index = np.where(AOLP_scores < avg_score)[0]
            
            if len(mask_index) == len(options) - 1:
                IE_answer = option_id[np.argmax(AOLP_scores)]
            else:
                IE_options = options.copy()
                IE_scores = np.zeros(len(IE_options))
                for index in mask_index:
                    IE_options[index] = '[MASK]'
                for j, option in enumerate(IE_options):
                    IE_prompt = create_prompt(question, IE_options)
                    IE_log_prob, IE_num_token = cal_score(IE_prompt, option)
                    IE_scores[j] = IE_log_prob
                IE_scores[mask_index] = -np.inf
                IE_answer = option_id[np.argmax(IE_scores)]
           
            path = args.output_path + f'/{d}/{model_name}/IE.txt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and i == 0:
                os.remove(path)
            with open (path, 'a') as f:
                f.write(f'{IE_answer}\n')

