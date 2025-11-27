import re
import gc
import json
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer

# ===============================
# Configuration
# ===============================
max_seq_length = 10000
lora_rank = 32

BASE_MODEL_PATH = "PATH/TO/BASE_MODEL"
TRAIN_DATA_PATH = "PATH/TO/TRAIN_DATA.jsonl"
SAVE_PATH = "PATH/TO/SAVE_MODEL"


# ===============================
# Model Loading
# ===============================
def load_model(model_path: str):
    """Load base model with LoRA enabled."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer


# ===============================
# JSONL Loader
# ===============================
def load_jsonl(file_path: str) -> pd.DataFrame:
    """Load a JSONL file into a pandas DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


# ===============================
# Timeseries Processing
# ===============================
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


# ===============================
# Construct Chat Messages
# ===============================
def format_sample(row, system_prompt):
    """Format a dataset row into a chat message structure."""
    cot = row.get("cot")
    problem = row.get("updated_question")

    if not isinstance(cot, str):
        print(f"[Warning] CoT is not a string. Type={type(cot)} Value={cot}")
        cot = "No reasoning provided"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": cot},
    ]


# ===============================
# Main Training Pipeline
# ===============================
def main():
    # Load model
    model, tokenizer = load_model(BASE_MODEL_PATH)

    # Build system prompt
    reasoning_start = "<THINK>"
    reasoning_end = "</THINK>"
    solution_start = "<ANSWER>"
    solution_end = "</ANSWER>"

    system_prompt = (
        f"You are given a time series reasoning problem.\n"
        f"Think about the problem and provide your reasoning between {reasoning_start}{reasoning_end}.\n"
        f"Then provide the final answer between {solution_start}{solution_end}."
    )

    print("Loaded system prompt:\n", system_prompt)

    # Load training data
    dataset = load_jsonl(TRAIN_DATA_PATH)

    # Drop unused fields
    if "timeseries" in dataset.columns:
        print("Dropping column 'timeseries' ...")
        dataset = dataset.drop(columns=["timeseries"])

    # Preprocess questions
    dataset["updated_question"] = dataset.apply(process_timeseries, axis=1)

    # Drop the now-unused timeseries2
    if "timeseries2" in dataset.columns:
        dataset = dataset.drop(columns=["timeseries2"])

    # Format chat messages
    dataset["Messages"] = dataset.apply(
        lambda x: format_sample(x, system_prompt), axis=1
    )

    # Create text field from tokenizer chat template
    dataset["text"] = tokenizer.apply_chat_template(
        dataset["Messages"].tolist(), tokenize=False
    )

    hf_dataset = Dataset.from_pandas(dataset)

    print("Dataset preview:")
    print(hf_dataset)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )

    trainer.train()

    # Save model
    print("Saving model...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Model saved to:", SAVE_PATH)


# ===============================
# Run Script
# ===============================
if __name__ == "__main__":
    main()
