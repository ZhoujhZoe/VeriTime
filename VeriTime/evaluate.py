from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
import json
import jsonlines
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import numpy as np
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
import gc
import re
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer


# Parameters used during training
max_seq_length = 10000
lora_rank = 32

reasoning_start = "<THINK>"
reasoning_end   = "</THINK>"
solution_start  = "<ANSWER>"
solution_end    = "</ANSWER>"


system_prompt = \
f"""You are given a time series reasoning problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    # SFT model path
    model_name = "PATH/TO/SFT_MODEL",
    # GRPO model path (optional)
    # model_name = "PATH/TO/GRPO_MODEL",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7,
)

chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")

tokenizer.chat_template = chat_template


def process_timeseries(row):
    question = row['question']
    timeseries2 = row['timeseries2']

    ts_str = timeseries2

    # ts_data = timeseries2[0] if isinstance(timeseries2[0], list) else timeseries2
    # ts_str = ', '.join(map(str, ts_data))

    # Replace <ts><ts/> placeholder with raw time series
    updated_question = question.replace('<ts><ts/>', ts_str)
    return updated_question


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def evaluate_model():
    # Load test dataset
    dataset = load_jsonl("/data/home/jiahui/TSRLVRReasoning/zoe/data/Timebed/timebed_v1_test_558.jsonl")

    columns_to_drop = ['timeseries']  # Columns to remove
    for col in columns_to_drop:
        if col in dataset.columns:
            print(f"Removing column '{col}'...")
            dataset = dataset.drop(columns=[col])

    # Evaluation range
    test_data = dataset.iloc[:300]
    # test_data = dataset

    test_data["updated_question"] = test_data.apply(process_timeseries, axis=1)

    # Output path
    output_file = 'PATH/TO/TEST_DATA.jsonl'

    with jsonlines.open(output_file, mode='w') as writer:
        for index, item in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating model"):
            test_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['updated_question']}
            ]

            # Build prompt
            text = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(text, return_tensors="pt").to("cuda")

            output = model.generate(
                **inputs,
                temperature=0.2,  # generation temperature
                max_new_tokens=max_seq_length,
                streamer=TextStreamer(tokenizer, skip_prompt=False),
            )

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
            # Extract only model output (remove input part)
            input_length = len(text)
            llm_output = generated_text[input_length:].strip()

            # Prepare output JSONL record
            output_data = {
                "id": item["id"],
                "task": item["task"],
                "question": item["question"],
                "llm_output": llm_output,
                "label": item["label"],
                "step6_label": item["step6_label"]
            }
                
            writer.write(output_data)

            del inputs, output, generated_text
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    evaluate_model()
