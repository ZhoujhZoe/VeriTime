import json
import re
import time
from openai import OpenAI

"""conda environment: rebuttal"""

def classify_ts_task(input_text, output_text) -> int:
    """
    Classify time series tasks based on input text.
    Task definitions:
    1. Anomaly detection (class 1): contains anomaly-related keywords and is a True/False judgment about TS/abnormality.
    2. Scenario attribution (class 2): contains "choose from", representing multi-choice attribution or forecasting tasks.
    3. Inferential calculation (class 3): contains expressions like "how many", focusing on counting events/phenomena in TS.
    4. Others (class 4): does not satisfy the above categories.
    Return: task category ID (1/2/3/4)
    """
    # Case-insensitive matching
    input_lower = input_text.lower()
    output_lower = output_text.lower()
    
    # 1. Scenario attribution
    if re.search(r'\bchoose\b\s+\bfrom\b', input_lower):
        return 2
    
    # 2. Inferential calculation ("how many" and extended forms)
    if re.search(r'\bhow\b\s+\bmany\b', input_lower):
        return 3
    
    # 3. Anomaly detection: contains anomaly-related keywords + yes/no decision logic
    anomaly_keywords = {
        "normal", 
        "abnormal", "anomalous", "anomaly", "anomalies", 
        "usual", "unusual", 
        "expected", "unexpected",
        "extreme",
    }
    # Check if any anomaly keyword exists
    has_anomaly_keyword = any(
        re.search(r'\b' + re.escape(keyword) + r'\b', input_lower) 
        for keyword in anomaly_keywords
    )
    # Check output contains yes/no
    output_contains_yes_no = re.search(r'\byes\b', output_lower) or re.search(r'\bno\b', output_lower)
    
    if has_anomaly_keyword and output_contains_yes_no:
        return 1
    
    # 4. Others (class 4)
    return 4


def process_data(input_file, univariate_out_file, multivariate_out_file, start_idx, end_idx):
    open(univariate_out_file, 'w').close()
    open(multivariate_out_file, 'w').close()
    
    with open(univariate_out_file, 'a') as f_uni, open(multivariate_out_file, 'a') as f_multi:
        with open(input_file, 'r') as f_in:
            for idx, line in enumerate(f_in):
                if idx < start_idx:
                    continue
                if idx > end_idx:
                    break
                                        
                try:
                    data = json.loads(line.strip())
                    input_text = data.get("input", "")
                    output_text = data.get("output", "")
                    
                    # Rule-based classification
                    category = classify_ts_task(input_text, output_text)
                                       
                    if category == 4:
                        print(f"ID {idx}: classified as 4 (others), skipped")
                        continue
                    
                    # Count number of <ts> tags
                    ts_count = len(re.findall(r'<ts><ts/>', input_text))
                    
                    # Map category to task type
                    task_map = {
                        1: "Anomaly detection",
                        2: "Scenario attribution",
                        3: "Inferential calculation"
                    }
                    
                    # Build output object
                    output_data = {
                        "id": idx,
                        "task": task_map[category],
                        "question": input_text,
                        "output": data["output"],
                        "label": "",
                        "timeseries": data["timeseries"]
                    }
                     
                    if ts_count == 1:
                        f_uni.write(json.dumps(output_data) + '\n')
                        print(f"ID {idx}: written to univariate.json (category: {category})")
                    elif ts_count >= 2:
                        f_multi.write(json.dumps(output_data) + '\n')
                        print(f"ID {idx}: written to multivariate.json (category: {category}, TS count: {ts_count})")
                    else:
                        print(f"ID {idx}: no <ts> tag found")
                        
                except json.JSONDecodeError:
                    print(f"ID {idx}: JSON parsing error")
                except KeyError as e:
                    print(f"ID {idx}: missing required field - {e}")
                except Exception as e:
                    print(f"ID {idx}: processing error - {e}")
    

if __name__ == "__main__":
    input_file = "./sft/chatts_sft_train.jsonl"
    start_index = 0  # Start index (inclusive)
    end_index = 50000    # End index (inclusive)
    
    univariate_out_file = 'univariate_rule_based.jsonl'
    multivariate_out_file = 'multivariate_rule_based.jsonl'

    process_data(input_file, univariate_out_file, multivariate_out_file, start_index, end_index)
