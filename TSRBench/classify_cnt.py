import re
import json
from typing import List, Dict
from word2number import w2n 


def process_jsonl_label(input_file: str) -> None:
    anomaly_cnt = 0
    scenario_cnt = 0
    inferential_cnt = 0
    wrong_id = []
    total_cnt = 0

    with open(input_file, 'r', encoding="utf-8") as f_in:
        for idx, line in enumerate(f_in):

            # Skip empty lines
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line.strip())
                
                id = data["id"]
                task = data["task"].strip()
                
                # Task classification
                if task == "Anomaly detection":
                    anomaly_cnt += 1
                elif task == "Scenario attribution":
                    scenario_cnt += 1
                elif task == "Inferential calculation":
                    inferential_cnt += 1
                else:
                    print(f"ID {id}: Unknown task type '{task}', skipped")
                    wrong_id.append(id)
                    continue
    
                print(f"ID {id}: Task = {task}")
                total_cnt += 1
            
            except json.JSONDecodeError as e:
                print(f"Error: ID {id} failed to process - {str(e)}, skipped")
            except KeyError as e:
                print(f"Error: ID {id} missing field - {str(e)}, skipped")
            
        print(f"\nSummary of task distribution:")
        print(f"Total samples: {total_cnt}")
        print(f"Anomaly detection: {anomaly_cnt}")
        print(f"Scenario attribution: {scenario_cnt}")
        print(f"Inferential calculation: {inferential_cnt}")
        print(f"Failed sample IDs: {wrong_id}")
    


if __name__ == "__main__":

    input_path = "./univariate_0_2000_filtered_labeled_cot_stepLabeled_correct_step2label.jsonl"

    # Run processing
    process_jsonl_label(input_path)
