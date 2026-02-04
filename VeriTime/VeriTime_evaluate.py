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


# Define a configuration dictionary
config = {
    "max_seq_length": 10000,  # Or 8192
    "lora_rank": 32,
    "model_name": "PATH/TO/FINETUNED_MODEL",
    "dataset": "PATH/TO/DATASET.jsonl",
    "output_file": "PATH/TO/OUTPUT_FILE.jsonl",
    "temperature": 0.2
}


reasoning_start = "<THINK>" 
reasoning_end   = "</THINK>"   
solution_start  = "<ANSWER>"
solution_end    = "</ANSWER>"


system_prompt = \
f"""You are given a time series reasoning problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["model_name"],
    max_seq_length=config["max_seq_length"],
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = config["lora_rank"],
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

    # If 'timeseries2' is not in the row, return the question as is
    if 'timeseries2' not in row:
        return question

    timeseries2 = row['timeseries2']
    task = row['task']

    if task in ["Anomaly detection", "Scenario attribution", "Inferential calculation"]:
        # Processing for ChatTS-style data
        ts_data = timeseries2[0] if isinstance(timeseries2[0], list) else timeseries2
        ts_str = ', '.join(map(str, ts_data))
    else:
        # Processing for TimeBed-style data
        ts_str = timeseries2
    
    # Replace the <ts><ts/> placeholder in the question
    updated_question = question.replace('<ts><ts/>', ts_str)
    return updated_question


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def evaluate_model():
    # Load dataset from config
    dataset = load_jsonl(config["dataset"])
   
    columns_to_drop = ['timeseries']  # Add other unnecessary columns
    for col in columns_to_drop:
        if col in dataset.columns:
            print(f"Deleting column '{col}'...")
            dataset = dataset.drop(columns=[col])
   
    # Use the dataset
    test_data = dataset
    # test_data = dataset.iloc[378:] # Example of slicing the dataset

    test_data["updated_question"] = test_data.apply(process_timeseries, axis=1)

    # Print the 'updated_question' of the first case
    print("First case 'updated_question':", test_data.iloc[0]['updated_question'])

    # Use output file path from config
    with jsonlines.open(config["output_file"], mode='w') as writer:
        for index, item in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating model"):
            test_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['updated_question']}
            ]

            # Generate input text
            text = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(text, return_tensors="pt").to("cuda")

            # Use temperature from config
            output = model.generate(
                **inputs,
                temperature=config["temperature"],
                max_new_tokens=config["max_seq_length"],
                streamer=TextStreamer(tokenizer, skip_prompt=False),
            )


            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
            # Extract model output (remove input)
            input_length = len(text)
            llm_output = generated_text[input_length:].strip()

            # Prepare output data
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