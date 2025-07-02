#!/bin/bash

model_path=(
    'LLM/Qwen2.5-14B'
    'LLM/Qwen2.5-32B'
    'LLM/gemma-2-9b'
    'LLM/gemma-2-9b-it'
    'LLM/gemma-3-12b-pt'
    'LLM/gemma-3-27b-pt'
)

task=(
    'option_id'
    'ablation'
    'TG'
)

for m in "${model_path[@]}"; do
    for t in "${task[@]}"; do

        nohup python src/$t.py --model_path="$m" --cuda_device="0" --output_path="main_result"
        wait
    done
done
