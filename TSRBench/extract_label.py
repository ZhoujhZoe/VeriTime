import re
import json
from typing import List, Dict
from word2number import w2n 


def extract_anomaly_label(output: str) -> str | None:
    """Anomaly detection: extract Yes/No (priority) or Normal/Abnormal (secondary)"""
    
    # Handle non-string input; remove extra whitespace/newlines
    output_clean = output.strip().replace("\n", " ") if isinstance(output, str) else ""
    if not output_clean:
        return None
    
    yes_no_pattern = r"(?i)\b(Yes|No)\b"
    normal_abnormal_pattern = r"(?i)\b(Normal|Abnormal)\b"

    # Extract Yes/No first
    yes_no_match = re.search(yes_no_pattern, output_clean)
    if yes_no_match:
        return yes_no_match.group().strip().capitalize()

    # Extract Normal/Abnormal second
    na_match = re.search(normal_abnormal_pattern, output_clean)
    if na_match:
        return na_match.group().strip().capitalize()

    return None


def extract_scenario_label(output: str) -> str | None:
    """Scenario attribution: extract the first sentence"""
    output_clean = output.strip().replace("\n", " ").replace("\r", " ") if isinstance(output, str) else ""
    if not output_clean:
        return None
    
    first_sentence_pattern = r"^.*?[.!?]"
    sentence_match = re.search(first_sentence_pattern, output_clean)
    
    if sentence_match:
        first_sentence = sentence_match.group().rstrip(".!?").strip()
        return first_sentence if first_sentence else None
    else:
        # If no punctuation, return the whole non-empty content
        return output_clean.strip() if output_clean.strip() else None


def extract_inferential_label(output: str) -> str | None:
    """Inferential calculation: extract counting result (supports English words & digits)"""
    output_clean = output.strip().replace("\n", " ") if isinstance(output, str) else ""
    if not output_clean:
        return None

    # Regex patterns covering common inferential counting structures
    infer_patterns = [
        r"I've found that there are (\w+|\d+)",
        r"I've found that there is (\w+|\d+)",
        r"I've found that there were (\w+|\d+)",
        r"I've found that there was (\w+|\d+)",
        r"I've found (\w+|\d+)",
        r"I've identified (\w+|\d+)",
        r"there is (\w+|\d+)",
        r"there are (?:approximately|about|roughly) (\w+|\d+)",
        r"there was (\w+|\d+)",
        r"there were (?:approximately|about|roughly) (\w+|\d+)",
        r"it was observed that (\w+|\d+)",
        r"the number of .*? is (\w+|\d+)",
        r"it took (\w+|\d+)",
        r"It took (\w+|\d+)",
        r"(\w+|\d+) \w+(?: \w+)* can be identified",

        # Lower priority patterns
        r"occurred (\w+|\d+)",
        r"the time series shows (\w+|\d+)",
        r"(\w+|\d+) times\b",
        r"(\w+|\d+) day\b",
        r"(\w+|\d+) days\b",
        r"(\w+|\d+) minute\b",
        r"(\w+|\d+) minutes\b",
        r"(\w+|\d+) hour\b",
        r"(\w+|\d+) hours\b",
        r"(\w+|\d+) second\b",
        r"(\w+|\d+) seconds\b",
        r"(\w+|\d+) point\b",
        r"(\w+|\d+) points\b",
        r"on (\w+|\d+)", 
    ]

    special_map = {
        "no": "0",
        "zero": "0",
        "none": "0",
        "a": "1",
        "an": "1",
        "once": "1",
        "twice": "2"
    }

    # Iterate through patterns
    for pattern in infer_patterns:
        match = re.search(pattern, output_clean, re.IGNORECASE)
        if match:
            num_raw = match.group(1).strip()
            
            # Return direct digits
            if num_raw.isdigit():
                return num_raw
            
            # Handle special keywords
            if num_raw in special_map:
                return special_map[num_raw]

            try:
                num_digit = w2n.word_to_num(num_raw)
                return str(num_digit)
            except ValueError:
                # Fallback: return raw string when word-number is malformed
                return num_raw
            
    return None


# Custom exception for non-numeric timeseries values
class NonNumericValueError(Exception):
    pass

# Custom exception for empty extracted label
class EmptyLabelError(Exception):
    pass


def round_timeseries_values(timeseries):
    """Recursively process timeseries list and round all numeric values to 4 decimals"""
    processed = []
    for item in timeseries:
        if isinstance(item, list):
            processed.append(round_timeseries_values(item))  # Recursive for nested lists
        elif isinstance(item, (int, float)):
            processed.append(round(item, 4))  # Keep 4 decimals
        else:
            # Raise error for non-numeric data
            raise NonNumericValueError(
                f"Non-numeric value found in timeseries: {item} (type: {type(item).__name__})"
            )
    return processed


def process_jsonl_label(input_file: str, output_file: str, start_idx: int, end_idx: int) -> None:
    # Mapping task types to extraction functions
    task_to_extractor = {
        "Anomaly detection": extract_anomaly_label,
        "Scenario attribution": extract_scenario_label,
        "Inferential calculation": extract_inferential_label
    }
    
    wrong_id = []  # Record IDs that fail to process for manual inspection
    with open(output_file, 'a', encoding="utf-8") as f_out:
        with open(input_file, 'r', encoding="utf-8") as f_in:
            for idx, line in enumerate(f_in):
                if idx < start_idx:
                    continue
                if idx > end_idx:
                    break
                
                # Skip empty lines
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line.strip())
                    
                    id = data["id"]
                    task = data["task"].strip()
                    output = data["output"]
                    timeseries = data["timeseries"]
                    
                    if task not in task_to_extractor:
                        print(f"ID {id}: Unknown task type {task}, skipped")
                        continue
                    
                    # Extract label using corresponding function
                    extractor = task_to_extractor[task]
                    label = extractor(output)

                    if label is None:
                        print(f"ID {id}: Task '{task}' extracted an empty label")
                        wrong_id.append(id)

                    if task == "Inferential calculation" and not str(label).isdigit():
                        print(f"ID {id}: Task {task}, extracted non-numeric label: {data['label']}")
                        wrong_id.append(id)
                    
                    # Round all timeseries numeric values to 4 decimals
                    data["timeseries2"] = round_timeseries_values(timeseries)
                    
                    data["label"] = label if label is not None else ""
                    
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    print(f"ID {id}: Task {task}, extracted label: {data['label']}")
                
                except json.JSONDecodeError as e:
                    print(f"ID {id}: JSON parsing error {str(e)}, skipped")
                    wrong_id.append(id)
                except Exception as e:
                    print(f"ID {id}: Processing error {str(e)}, skipped")
                    wrong_id.append(id)
                
            print(f"Failed: {len(wrong_id)} samples. Failed IDs: {wrong_id}")
            print("All timeseries values rounded to 4 decimals and saved in 'timeseries2'")
    


if __name__ == "__main__":

    input_path = "./univariate_classified_2001_6000.jsonl"
    output_path = "./univariate_classified_2001_6000_testttt.jsonl"
    
    start_index = 0  # Start index (inclusive)
    end_index = 1300  # End index (inclusive)
    
    # Clear output file
    open(output_path, 'w').close()

    # Execute batch processing
    process_jsonl_label(input_path, output_path, start_index, end_index)
