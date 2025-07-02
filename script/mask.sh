#!/bin/bash

model_path=(
    'LLM/Qwen2.5-14B'
    'LLM/gemma-3-12b-pt'
)

task=(
    'mask'
)

for m in "${model_path[@]}"; do
    for t in "${task[@]}"; do

        nohup python src/$t.py --model_path="$m" --cuda_device="0" --output_path="mask_result"
        wait
    done
done
