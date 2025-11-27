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
import string
from fuzzywuzzy import fuzz

max_seq_length = 10000  
lora_rank = 32  

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "PATH/TO/BASE_MODEL",  # SFT fine-tuned model path
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,  
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank * 2,  
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
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

print("System prompt:", system_prompt)

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

chat_template = chat_template \
    .replace("'{system_prompt}'",   f"'{system_prompt}'") \
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")

tokenizer.chat_template = chat_template


# Load JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Load dataset
dataset = load_jsonl("PATH/TO/TRAIN_DATA.jsonl")

# Drop unnecessary columns
columns_to_drop = ['timeseries']
for col in columns_to_drop:
    if col in dataset.columns:
        print(f"Dropping column '{col}'...")
        dataset = dataset.drop(columns=[col])

# Modify dataset sample range
# dataset = dataset.iloc[100:700]

def process_timeseries(row):
    """Replace <ts><ts/> placeholder with serialized time-series."""
    question = row["question"]
    timeseries2 = row["timeseries2"]
    task = row["task"]

    if task in ["Anomaly detection", "Scenario attribution", "Inferential calculation"]:
        ts_data = timeseries2[0] if isinstance(timeseries2[0], list) else timeseries2
        ts_str = ", ".join(map(str, ts_data))
    else:
        ts_str = str(timeseries2)

    updated = question.replace("<ts><ts/>", ts_str)
    return updated


dataset["updated_question"] = dataset.apply(process_timeseries, axis=1)

dataset = Dataset.from_pandas(dataset)

print(dataset)

# Construct final dataset format
original_columns = dataset.column_names
dataset = dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x["updated_question"]},
    ],
    "answer": x["label"],
    "step1_label": x["step1_label"],
    "step2_label": x["step2_label"],
    },
    remove_columns=original_columns
)
print(dataset)


############### Reward functions ###############

solution_end_regex = r"</ANSWER>[\s]{0,}" + \
    "(?:" + re.escape(tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end_regex}"\
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
)

# Check whether response matches required structure
def match_format_exactly(completions, **kwargs):
    scores = []

    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]

        print(f"\n--- Response {i+1} ---")
        print(f"Full response: {response}")

        if match_format.search(response) is not None:
            score += 3.0
        else:
            score -= 2.0

        scores.append(score)
        print(f"Final score: {score}")

    return scores


# Reward based on output length
def length_reward(completions, **kwargs):
    scores = []
    max_length = 3500
    threshold_length = 800

    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]

        tokens = tokenizer.encode(response, add_special_tokens=False)
        length = len(tokens)

        print(f"\n--- Response {i+1} ---")
        print(f"Token length: {length}")

        if threshold_length < length < max_length:
            score += 1.0
        elif length <= threshold_length:
            score += 1.0 * (length / max_length)
        else:
            score -= 2.0

        scores.append(score)
        print(f"Final score: {score}")

    return scores


# Reward final answer correctness
def check_final_answer(completions, answer, **kwargs):
    scores = []
    answer_pattern = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL)
    answer_content_pattern = re.compile(r"The answer is\s*(.+)\.?", re.IGNORECASE)

    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]

        print(f"\n--- Response {i+1} ---")
        print(f"Label: {answer[i]}")

        match = answer_pattern.search(response)
        if match:
            answer_content = match.group(1).strip()
            print(f"Answer content: '{answer_content}'")

            content_match = answer_content_pattern.search(answer_content)
            if content_match:
                extracted = content_match.group(1).strip()
                print(f"Extracted: '{extracted}'")

                if answer[i].lower() in extracted.lower():
                    score += 5.0
                else:
                    score -= 2.0
            else:
                if answer[i].lower() in answer_content.lower():
                    score += 5.0
                else:
                    score -= 2.0
        else:
            score -= 2.0

        scores.append(score)
        print(f"Final score: {score}")

    return scores


def remove_trailing_period(phrase):
    return phrase.rstrip('.').strip()


# Reward intermediate steps (process correctness)
def check_process_answer(completions, answer, step1_label, step2_label, **kwargs):

    scores = []

    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]

        print(f"\n--- Response {i+1} ---")
        print(f"Full response: {response}")

        think_content = response

        ### Step 1
        pattern1 = re.compile(r"Step 1.*?\[Judgment\](.*?)(?=\[Description\]|Step 2|$)",
                              re.DOTALL | re.IGNORECASE)
        match1 = pattern1.search(think_content)
        if match1:
            judgment1 = match1.group(1).strip()
            label_text = step1_label[i].lower()
            judgment_text = judgment1.lower()
            print(f"Step 1 Judgment: '{judgment1}'")
            print(f"Step 1 Label: '{step1_label[i]}'")

            if label_text in judgment_text or judgment_text in label_text:
                score += 1.0
            elif fuzz.partial_ratio(label_text, judgment_text) > 80:
                score += 1.0
        else:
            print("Step 1 judgment not found.")

        ### Step 2
        pattern2 = re.compile(r"Step 2.*?\[Judgment\](.*?)(?=\[Description\]|Step 3|$)",
                              re.DOTALL | re.IGNORECASE)
        match2 = pattern2.search(think_content)
        if match2:
            judgment2 = match2.group(1).strip()
            judgment_phrases = [remove_trailing_period(p).lower()
                                for p in judgment2.split(';') if p.strip()]
            label_phrases = [remove_trailing_period(p).lower()
                             for p in step2_label[i].split(';') if p.strip()]

            matched_count = 0
            threshold = 90

            for jp in judgment_phrases:
                best = max(fuzz.partial_ratio(jp, lp) for lp in label_phrases)
                if best >= threshold:
                    matched_count += 1

            if matched_count >= 2:
                score += 2.0
            elif matched_count == 1:
                score += 1.0

        ### Step 4
        pattern4 = re.compile(r"Step 4.*?\[Judgment\](.*?)(?=\[Description\]|Step 5|$)",
                              re.DOTALL | re.IGNORECASE)
        match4 = pattern4.search(think_content)
        if match4:
            judgment4 = match4.group(1).strip()
            if answer[i].lower() in judgment4.lower():
                score += 2.0

        ### Step 6
        pattern6 = re.compile(r"Step 6.*?\[Judgment\](.*?)(?=\[Description\]|<ANSWER>|$)",
                              re.DOTALL | re.IGNORECASE)
        match6 = pattern6.search(think_content)
        if match6:
            judgment6 = match6.group(1).strip()
            if answer[i].lower() in judgment6.lower():
                score += 1.0

        scores.append(score)
        print(f"Process score: {score}")

    return scores


#############################################
# Tokenization & filtering

tokenized = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(
        x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=False,
)
print(tokenizer.decode(tokenized[0]["tokens"]))

tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

lengths = tokenized["L"]
print(f"Mean: {np.mean(lengths)}")
print(f"Median: {np.median(lengths)}")
print(f"80% quantile: {np.quantile(lengths, 0.8)}")
print(f"85% quantile: {np.quantile(lengths, 0.85)}")
print(f"90% quantile: {np.quantile(lengths, 0.9)}")
print(f"95% quantile: {np.quantile(lengths, 0.95)}")

maximum_length = 6190
print("Max Length =", maximum_length)

dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
print(f"Filtered dataset size: {len(dataset)} samples")
del tokenized

max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length
print("max_prompt_length", max_prompt_length)
print("max_completion_length", max_completion_length)


vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

# Training configuration
training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=0.8,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_train_epochs=1,
    save_steps=647,
    report_to="none",
    output_dir="PATH/TO/SAVE_MODEL",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_format_exactly,
        length_reward,
        check_final_answer,
        check_process_answer,
    ],
    args=training_args,
    train_dataset=dataset,
    max_grad_norm=1.0,
)

trainer.train()
