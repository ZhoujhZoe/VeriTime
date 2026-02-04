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

# General configuration
class GeneralConfig:
    max_seq_length = 10000 # Or 8192
    lora_rank = 32
    model_name = "PATH/TO/FINETUNED_MODEL"
    dataset_path = "PATH/TO/DATASET.jsonl" 
    output_dir = "./outputs/my_grpo_model/"
    random_state = 3407
    maximum_length = 6190

general_config = GeneralConfig()

# Training configuration
class TrainingConfig:
    num_train_epochs = 1
    max_steps = 20  # Set to None if not used
    save_steps = 1545
    temperature = 0.8
    learning_rate = 5e-6
    seed = 3407
    per_device_train_batch_size = 4
    num_generations = 4

training_config = TrainingConfig()

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = general_config.model_name,
    max_seq_length = general_config.max_seq_length,
    max_lora_rank = general_config.lora_rank,
    load_in_4bit = False,
    fast_inference = True,
    gpu_memory_utilization = 0.7,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = general_config.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = general_config.lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = general_config.random_state,
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

# Load the dataset
dataset = load_jsonl(general_config.dataset_path)

columns_to_drop = ['timeseries']  # Add other columns to drop if needed
for col in columns_to_drop:
    if col in dataset.columns:
        print(f"Dropping column '{col}'...")
        dataset = dataset.drop(columns=[col])

# Slice the dataset
dataset = dataset.iloc[100:700]


def process_timeseries(row):
    question = row['question']
    timeseries2 = row['timeseries2']
    task = row['task']
    
    if task in ["Anomaly detection", "Scenario attribution", "Inferential calculation"]:
        ts_data = timeseries2[0] if isinstance(timeseries2[0], list) else timeseries2
        ts_str = ', '.join(map(str, ts_data))
    else:
        ts_str = timeseries2
    
    # Replace the <ts><ts/> placeholder in the question
    updated_question = question.replace('<ts><ts/>', ts_str)
    return updated_question

dataset["updated_question"] = dataset.apply(process_timeseries, axis=1)
dataset = dataset.drop(columns='timeseries2')

dataset = Dataset.from_pandas(dataset)

print(dataset)


# Process the dataset
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
    remove_columns=original_columns  # Remove all original columns
)
print(dataset)
# print(dataset[0])


############### Custom reward functions start here

# Regex to match </ANSWER> and any following characters
solution_end_regex = r"</ANSWER>[\s]{0,}" + \
    "(?:" + re.escape(tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end_regex}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

# Reward function for exact format matching
def match_format_exactly(completions, **kwargs):
    scores = []

    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]

        # print(f"\n--- Response {i+1} ---")
        # print(f"Full response: {response}")

        # Match if format is seen exactly!
        if match_format.search(response) is not None: 
            score += 3.0
        else:
            score -= 2.0

        scores.append(score)

        # print(f"Final score for response {i+1}: {score}")

    return scores


# Reward function for output length
def length_reward(completions, **kwargs):
    scores = []
    max_length = 3500   # Example for a specific dataset
    threshold_length = 800

    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]

        # Get the number of tokens in the response
        tokens = tokenizer.encode(response, add_special_tokens=False)
        length = len(tokens)

        # print(f"\n--- Response {i+1} ---")
        # print(f"Token length: {length}")

        if threshold_length < length < max_length:
            score += 1.0
        elif length <= threshold_length:
            score += 1.0 * (length / max_length)
        else:
            score -= 2.0

        scores.append(score)

        # print(f"Final score for response {i+1}: {score}")

    return scores


# Reward function for the final answer
def check_final_answer(completions, answer, **kwargs):
    scores = []
    answer_pattern = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL)
    answer_content_pattern = re.compile(r"The answer is\s*(.+)\.?", re.IGNORECASE)


    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]

        print(f"\n--- Response {i+1} ---")
        print(f"Label: {answer[i]}")

        # Try to extract content from the <ANSWER> tag
        answer_match = answer_pattern.search(response)
        if answer_match:
            answer_content = answer_match.group(1).strip()

            # Try to extract content after "The answer is"
            content_match = answer_content_pattern.search(answer_content)
            if content_match:
                extracted_answer = content_match.group(1).strip()

                # Compare with the label (case-insensitive and ignoring surrounding spaces)
                if answer[i].lower() in extracted_answer.lower():
                    score += 5.0
                else:
                    score -= 2.0
            else:
                # If "The answer is" format is not found, check if the label is in the answer content
                if answer[i].lower() in answer_content.lower():
                    score += 5.0
                else:
                    score -= 2.0
        else:
            score -= 2.0


        scores.append(score)
        # print(f"Final score for response {i+1}: {score}")

    return scores


# Function to remove trailing punctuation
def remove_trailing_period(phrase):
    return phrase.rstrip('.').strip()


def check_process_answer(completions, answer, step1_label, step2_label, **kwargs):

    scores = []

    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]
        print(f"\n--- Response {i+1} ---")
        print(f"Full response: {response}")
        
        think_content = response
        # Extract Judgment from Step 1
        pattern1 = re.compile(r"Step 1.*?\[Judgment\](.*?)(?=\[Description\]|Step 2|$)", re.DOTALL | re.IGNORECASE)
        match1 = pattern1.search(think_content)
        if match1:
            judgment1 = match1.group(1).strip()
            label_text = step1_label[i].lower()
            judgment_text = judgment1.lower()
            # print(f"Step 1 Judgment: '{judgment1}'")
            # print(f"Step 1 Label: '{step1_label[i]}'")

            # Strategy 1: Bidirectional containment match
            if label_text in judgment_text or judgment_text in label_text:
                score += 1.0
                # print("Step 1 judgment - bidirectional containment matched. score+1.0")
            
            # Strategy 2: Similarity match
            elif fuzz.partial_ratio(label_text, judgment_text) > 80:
                score += 1.0
                print(f"Step 1 judgment - similarity matched ({fuzz.partial_ratio(label_text, judgment_text)}). score+1.0")
            
            else:
                print("Step 1 judgment mismatched. score+0.0")
                
        else:
            print("Step 1 judgment not found. score+0.0")

        # Extract Judgment from Step 2
        pattern2 = re.compile(r"Step 2.*?\[Judgment\](.*?)(?=\[Description\]|Step 3|$)", re.DOTALL | re.IGNORECASE)
        match2 = pattern2.search(think_content)
        if match2:
            judgment2 = match2.group(1).strip()

            judgment_phrases = [remove_trailing_period(phrase).lower() 
                for phrase in judgment2.split(';') if phrase.strip()]

            label_phrases = [remove_trailing_period(phrase).lower() 
                            for phrase in step2_label[i].split(';') if phrase.strip()]
            
            # print(f"Step 2 Judgment phrases: {judgment_phrases}")
            # print(f"Step 2 Label phrases: {label_phrases}")

            # Calculate the number of matched phrases
            matched_count = 0
            similarity_threshold = 90

            for j_phrase in judgment_phrases:
                best_score = 0
                for l_phrase in label_phrases:
                    # Use partial_ratio for better partial matching
                    similarity_score = fuzz.partial_ratio(j_phrase, l_phrase)
                    if similarity_score > best_score:
                        best_score = similarity_score
            
                print(f"Phrase '{j_phrase}' best fuzzy score: {best_score}")
                if best_score >= similarity_threshold:
                    matched_count += 1
            
            # Award score based on the number of matches
            if matched_count >= 2:
                score += 2.0
            elif matched_count == 1:
                score += 1.0

            print(f"Matched count: {matched_count}")

        else:
            print("Step 2 judgment not found. score+0.0")
                
        # Extract Judgment from Step 4
        pattern4 = re.compile(r"Step 4.*?\[Judgment\](.*?)(?=\[Description\]|Step 5|$)", re.DOTALL | re.IGNORECASE)
        match4 = pattern4.search(think_content)
        if match4:
            judgment4 = match4.group(1).strip()
            # print(f"Step 4 Judgment: '{judgment4}'")
            # print(f"Label: '{answer[i]}'")

            if answer[i].lower() in judgment4.lower():
                score += 2.0
                print("Step 4 judgment matched. score+2.0")
            else:
                print("Step 4 judgment mismatched. score+0.0")
        else:
            print("Step 4 judgment not found. score+0.0")
        

        # Extract Judgment from Step 6
        pattern6 = re.compile(r"Step 6.*?\[Judgment\](.*?)(?=\[Description\]|<ANSWER>|$)", re.DOTALL | re.IGNORECASE)
        match6 = pattern6.search(think_content)
        if match6:
            judgment6 = match6.group(1).strip()
            # print(f"Step 6 Judgment: '{judgment6}'")
            # print(f"Label: '{answer[i]}'")

            if answer[i].lower() in judgment6.lower():
                score += 1.0
                print("Step 6 judgment matched. score+1.0")
            else:
                print("Step 6 judgment mismatched. score+0.0")
            
        else:
            print("Step 6 judgment not found. score+0.0")
    
        scores.append(score)
        print(f"Process score for response {i+1}: {score}")

    return scores



#############################################
# Dataset processing and tokenization
tokenized = dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = False, # Set to True for batch processing
)
print(tokenizer.decode(tokenized[0]["tokens"]))

tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

lengths = tokenized["L"]
print(f"Mean: {np.mean(lengths)}")
print(f"Median: {np.median(lengths)}") 
print(f"80th percentile: {np.quantile(lengths, 0.8)}") 
# print(f"85th percentile: {np.quantile(lengths, 0.85)}") 
# print(f"90th percentile: {np.quantile(lengths, 0.9)}") 
# print(f"95th percentile: {np.quantile(lengths, 0.95)}") 


# Filter dataset based on maximum length
dataset = dataset.select(np.where(np.array(tokenized["L"]) <= general_config.maximum_length)[0])
print(f"Number of samples after filtering: {len(dataset)}")
del tokenized

max_prompt_length = general_config.maximum_length + 1
max_completion_length = general_config.max_seq_length - max_prompt_length
print("max_prompt_length", max_prompt_length)
print("max_completion_length", max_completion_length) # e.g., 2000


# Sampling parameters
vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=training_config.seed,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

# Training arguments
training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=training_config.temperature,
    learning_rate=training_config.learning_rate,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=training_config.per_device_train_batch_size,
    gradient_accumulation_steps=1,
    num_generations=training_config.num_generations,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    ## Choose either num_train_epochs or max_steps
    num_train_epochs=training_config.num_train_epochs,
    # max_steps=training_config.max_steps,
    save_steps=training_config.save_steps,
    report_to="none",
    output_dir=general_config.output_dir,
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


