from unsloth import FastLanguageModel
import torch
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
import json

config = {
    "max_seq_length": 10000,  # 8192
    "lora_rank": 32,  # Larger rank = smarter, but slower
    "model_name": "PATH/TO/BASE_MODEL",  # Base model path
    "random_state": 3407,
    "dataset_path": "PATH/TO/DATASET.jsonl",
    "num_train_epochs": 1,  # Number of training epochs
    "learning_rate": 2e-4,  # Reduce to 2e-5 for long training runs
    "seed": 3407,
    "model_save_path": "PATH/TO/SAVE_MODEL",
}

max_seq_length = config["max_seq_length"] # 8192
lora_rank = config["lora_rank"] # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    # Modify the base model path
    model_name = config["model_name"],  
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
    # device_map = "balanced",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = config["random_state"],
)

reasoning_start = "<THINK>" 
reasoning_end   = "</THINK>"   
solution_start  = "<ANSWER>"
solution_end    = "</ANSWER>"

system_prompt = \
f"""You are given a time series reasoning problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

print("system prompt:", system_prompt)


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


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Modify the dataset
dataset = load_jsonl(config["dataset_path"])

columns_to_drop = ['timeseries']  # Add other columns to be dropped
for col in columns_to_drop:
    if col in dataset.columns:
        print(f"Deleting column '{col}'...")
        dataset = dataset.drop(columns=[col])

# Modify the dataset slice by id
# dataset = dataset.iloc[100:700]

def process_timeseries(row):
    question = row['question']
    timeseries2 = row['timeseries2']
    task = row['task']
     
    if task in ["Anomaly detection", "Scenario attribution", "Inferential calculation"]:
        # chatts processing method
        ts_data = timeseries2[0] if isinstance(timeseries2[0], list) else timeseries2
        ts_str = ', '.join(map(str, ts_data))
    else:
        # timebed processing method
        ts_str = timeseries2
    
    # Replace the <ts><ts/> placeholder in the question
    updated_question = question.replace('<ts><ts/>', ts_str)
    return updated_question

dataset["updated_question"] = dataset.apply(process_timeseries, axis=1)

dataset = dataset.drop(columns='timeseries2')


def format_dataset(x):
    try:
        cot = x["cot"]
        problem = x["updated_question"]

        # Check if cot is a string
        if not isinstance(cot, str):
            print(f"ERROR: cot is not string, type: {type(cot)}, value: {cot}")
            print(f"Row index: {x.name if hasattr(x, 'name') else 'unknown'}")
            # If it's NaN or other non-string types, it can be skipped or a default value can be used
            cot = "No reasoning provided"  # or return None to skip

        return [
            {"role" : "system",    "content" : system_prompt},
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : cot},
        ]
    except Exception as e:
        print(f"Error processing row: {x.name if hasattr(x, 'name') else 'unknown'}")
        print(f"Error message: {e}")
        print(f"cot value: {x.get('cot', 'Not found')}")
        print(f"problem value: {x.get('updated_question', 'Not found')}")
        raise



dataset["Messages"] = dataset.apply(format_dataset, axis = 1)

print("Sample formatted message:")
print(tokenizer.apply_chat_template(dataset["Messages"][0], tokenize=False))


dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
print(f"Maximum message length: {dataset['N'].max()} tokens")
print(f"Minimum message length: {dataset['N'].min()} tokens")
print(f"Average message length: {dataset['N'].mean():.2f} tokens")  

# tokenize the messages and convert it to a Hugging Face compatible dataset format:
dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize = False)
dataset = Dataset.from_pandas(dataset)
print(dataset)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = config["num_train_epochs"], 
        learning_rate = config["learning_rate"], # Reduce to 2e-5 for long training runs
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = config["seed"],
        report_to = "none", # Use this for WandB etc
    ),
)

trainer.train()

model.save_pretrained(config["model_save_path"])  # Save model
tokenizer.save_pretrained(config["model_save_path"])  # Save tokenizer