#!/bin/bash

model_path=(
    'LLM/Qwen2.5-14B'
    'LLM/gemma-3-12b-pt'
)

task=(
    'few_shot'
)

shot_number=(
    1
    10
)

for shots in "${shot_number[@]}"; do
    for m in "${model_path[@]}"; do
        for t in "${task[@]}"; do
        
            nohup python src/$t.py --model_path="$m" --cuda_device="0" --shots_number="$shots" --output_path="few_result"
            wait
        done
    done
done