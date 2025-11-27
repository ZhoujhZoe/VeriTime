import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import math
import os
import pandas as pd

def simple_ppl_split():
    # Initialize model and tokenizer
    model_name = "PATH/TO/MODEL"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    
    # Load data
    with open("PATH/TO/TRAIN_DATA.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    
    df = pd.DataFrame(data)
    
    # Calculate PPL for each case
    ppl_scores = []
    for idx, row in df.iterrows():
        # Create the text for PPL calculation (simplified version)
        text = f"Problem: {row['question']}\nSolution: {row['cot']}"
        
        # Calculate PPL
        encodings = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            ppl = math.exp(outputs.loss.item())
        
        ppl_scores.append((idx, ppl))
    
    # Sort by PPL (ascending - lower PPL is better)
    ppl_scores.sort(key=lambda x: x[1])
    
    # Split data
    best_indices = [idx for idx, ppl in ppl_scores[:1539]]
    remaining_indices = [idx for idx, ppl in ppl_scores[1539:]]
    
    # Save files
    os.makedirs("./ppl", exist_ok=True)
    
    with open("./ppl/chattsv1_sft_ppl.jsonl", 'w') as f:
        for idx in best_indices:
            f.write(json.dumps(data[idx], ensure_ascii=False) + '\n')
    
    with open("./ppl/chattsv1_grpo_ppl.jsonl", 'w') as f:
        for idx in remaining_indices:
            f.write(json.dumps(data[idx], ensure_ascii=False) + '\n')
    
    print(f"SFT subset: {len(best_indices)} cases")
    print(f"GRPO subset: {len(remaining_indices)} cases")

if __name__ == "__main__":
    simple_ppl_split()