import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import math
import os
import pandas as pd

class IFDEvaluator:
    def __init__(self, model_name="PATH/TO/MODEL", device="cuda"):
        """
        Initialize the IFD evaluator with a pre-trained causal LM.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        print(f"[Init] Loaded model {model_name} on {device}")

    def compute_nll(self, context: str, continuation: str) -> float:
        """
        Compute the average negative log-likelihood (NLL) of continuation given context.
        """
        # Encode context
        context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.model.device).long()
        # Encode continuation
        with self.tokenizer.as_target_tokenizer():
            cont_ids = self.tokenizer(continuation, return_tensors="pt").input_ids.to(self.model.device).long()

        # Concatenate input_ids
        input_ids = torch.cat([context_ids, cont_ids], dim=1)
        # Mask context for labels
        labels_ids = torch.cat([torch.full_like(context_ids, -100), cont_ids], dim=1)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=labels_ids)
            nll = outputs.loss.item()  # Average NLL

        return nll

    def compute_ifd(self, question: str, answer: str) -> float:
        """
        Compute IFD(Q, A) = s(A|Q) / s(A)
        """
        try:
            nll_cond = self.compute_nll(f"Q: {question}\nA:", answer)  # Conditional NLL
            nll_direct = self.compute_nll("", answer)  # Direct NLL
            return nll_cond / nll_direct
        except Exception as e:
            print(f"[Error] Failed IFD calculation: {e}")
            return float('inf')  # 使用正无穷表示失败

def simple_ifd_split():
    # Initialize IFD evaluator
    ifd_evaluator = IFDEvaluator()
    
    # Load data
    with open("PATH/TO/TRAIN_DATA.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    
    df = pd.DataFrame(data)
    
    # Calculate IFD for each case
    ifd_scores = []
    for idx, row in df.iterrows():
        # Extract question and answer (cot)
        question = row['question']
        answer = row['cot']
        
        # Calculate IFD
        ifd = ifd_evaluator.compute_ifd(question, answer)
        ifd_scores.append((idx, ifd))
    
    # Sort by IFD (ascending - lower IFD is better)
    ifd_scores.sort(key=lambda x: x[1])
    
    # Split data - select top 1539 cases with lowest IFD (highest quality)
    best_indices = [idx for idx, ifd in ifd_scores[:1539]]
    remaining_indices = [idx for idx, ifd in ifd_scores[1539:]]
    
    # Save files
    os.makedirs("./ifd", exist_ok=True)
    
    with open("./ifd/chattsv1_sft_ifd.jsonl", 'w') as f:
        for idx in best_indices:
            f.write(json.dumps(data[idx], ensure_ascii=False) + '\n')
    
    with open("./ifd/chattsv1_grpo_ifd.jsonl", 'w') as f:
        for idx in remaining_indices:
            f.write(json.dumps(data[idx], ensure_ascii=False) + '\n')
    
    # Print statistics
    valid_scores = [score for _, score in ifd_scores if score != float('inf')]
    if valid_scores:
        print(f"SFT subset: {len(best_indices)} cases (lowest IFD)")
        print(f"  - Lowest IFD: {ifd_scores[0][1]:.4f}")
        print(f"  - Highest IFD in SFT: {ifd_scores[1538][1]:.4f}")
        print(f"  - Average IFD in SFT: {sum(score[1] for score in ifd_scores[:1539] if score[1] != float('inf'))/1539:.4f}")
        
        print(f"GRPO subset: {len(remaining_indices)} cases (highest IFD)")
        if remaining_indices:
            print(f"  - Lowest IFD in GRPO: {ifd_scores[1539][1]:.4f}")
            print(f"  - Highest IFD: {ifd_scores[-1][1]:.4f}")
            print(f"  - Average IFD in GRPO: {sum(score[1] for score in ifd_scores[1539:] if score[1] != float('inf'))/len(remaining_indices):.4f}")
    else:
        print(f"SFT subset: {len(best_indices)} cases")
        print(f"GRPO subset: {len(remaining_indices)} cases")
        print("Warning: All IFD calculations failed!")

if __name__ == "__main__":
    simple_ifd_split()