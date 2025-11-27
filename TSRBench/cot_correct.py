import re
import json
from typing import List, Dict
from word2number import w2n 

def parse_cot_steps(cot_content: str) -> Dict[str, str | None]:
    """
    Parse the cot_deepseekr1 field and extract Step1~Step6 [Judgment] contents.
    Return format: {step1_label, step2_label, step4_label, step6_label}
    """
    step_labels = {
        "step1_label": None,
        "step2_label": None,
        "step4_label": None,
        "step6_label": None
    }
    if not isinstance(cot_content, str) or cot_content.strip() == "":
        return step_labels

    # Normalize whitespace to avoid formatting issues
    cot_clean = cot_content.strip().replace("\n", " ").replace("  ", " ")
    
    # Step 1: match [Judgment] → [Description]
    step1_pattern = r"Step 1.*?(?:\*\*)?\s*\[Judgment\]\s*(?:\*\*)?\s*([\s\S]+?)\s*(?:\*\*)?\s*\[Description\]\s*(?:\*\*)?"
    match = re.search(step1_pattern, cot_clean, re.IGNORECASE)
    if match:
        step_labels["step1_label"] = match.group(1).strip().replace("**", "").capitalize()

    # Step 2
    step2_pattern = r"Step 2.*?(?:\*\*)?\s*\[Judgment\]\s*(?:\*\*)?\s*([\s\S]+?)\s*(?:\*\*)?\s*\[Description\]\s*(?:\*\*)?"
    match = re.search(step2_pattern, cot_clean, re.IGNORECASE)
    if match:
        step_labels["step2_label"] = match.group(1).strip().replace("**", "").capitalize()

    # Step 4
    step4_pattern = r"Step 4.*?(?:\*\*)?\s*\[Judgment\]\s*(?:\*\*)?\s*([\s\S]+?)\s*(?:\*\*)?\s*\[Description\]\s*(?:\*\*)?"
    match = re.search(step4_pattern, cot_clean, re.IGNORECASE)
    if match:
        step_labels["step4_label"] = match.group(1).strip().replace("**", "").capitalize()

    # Step 6: match until the end (no Description section)
    step6_pattern = r"Step 6.*?(?:\*\*)?\s*\[Judgment\]\s*(?:\*\*)?\s*([\s\S]+?)\s*$"
    match = re.search(step6_pattern, cot_clean, re.IGNORECASE)
    if match:
        step6_tmp = match.group(1).strip().replace("**", "").capitalize()
    else:
        step6_tmp = None

    # Remove trailing punctuation
    if step6_tmp and isinstance(step6_tmp, str):
        step6_clean = re.sub(r'[.;]$', '', step6_tmp).strip()
    else:
        step6_clean = "unknown"
    step_labels["step6_label"] = step6_clean

    # Normalize values (remove empty indicators)
    for key in step_labels:
        if step_labels[key] in ["", "none", "null"]:
            step_labels[key] = None
            
    return step_labels


def extract_pure_number(text: str) -> int | None:
    """Extract a pure numeric value from text (supports units/symbols)."""
    if not isinstance(text, str):
        text = str(text)
    
    numeric_str = re.sub(r"[^\d.-]", "", text.strip())
    if not numeric_str or numeric_str in [".", "-", "-."]:
        return None
    
    try:
        return int(float(numeric_str))
    except ValueError:
        return None


def process_jsonl(input_file: str, correct_file: str, wrong_file: str) -> None:
    total_count = 0
    correct_count = 0
    wrong_count = 0
    error_count = 0
    error_id = []
    empty_label_id = []
    
    with open(correct_file, 'w', encoding="utf-8") as f_match, \
         open(wrong_file, 'w', encoding="utf-8") as f_mismatch:

        with open(input_file, 'r', encoding="utf-8") as f_in:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue
                total_count += 1

                try:
                    data = json.loads(line)

                    # Validate required fields
                    required_fields = ["id", "task", "output", "timeseries", "cot_deepseekr1", "label"]
                    for field in required_fields:
                        if field not in data:
                            raise KeyError(f"Missing required field: {field}")
                        
                    label = data["label"]
                    id = data["id"]
                    task = data["task"].strip()
                    cot_content = data["cot_deepseekr1"]
                    
                    # Extract step labels
                    step_labels = parse_cot_steps(cot_content)
                    
                    # Record empty step labels
                    for step, l in step_labels.items():
                        if l is None or l.strip() == "":
                            empty_label_id.append(id)
                            print(f" ID {id} : {step} is None or empty")

                    step6_label = step_labels.get("step6_label") or "unknown"

                    # Prepare new step fields
                    new_fields = {**step_labels}

                    # Build new dict and insert step fields after 'label'
                    new_data = {}
                    label_found = False
                    for key, value in data.items():
                        new_data[key] = value
                        if key == "label" and not label_found:
                            new_data.update(new_fields)
                            label_found = True
                    
                    if not label_found:
                        new_data.update(new_fields)
                        
                    # Normalize for comparison
                    def normalize_text(text: str) -> str:
                        return re.sub(r"[^\w\s]", "", text.strip().lower()).replace(" ", "")

                    norm_base = normalize_text(label)
                    norm_step6 = normalize_text(step6_label)

                    # Special rule: numeric comparison for 'Inferential calculation'
                    if task == "Inferential calculation":
                        step6_num = extract_pure_number(norm_step6)
                        base_num = extract_pure_number(norm_base)
                        is_match = (step6_num is not None and base_num is not None and step6_num == base_num)
                    else:
                        is_match = norm_step6 in norm_base or norm_base in norm_step6

                    # Output results
                    if is_match:
                        f_match.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                        correct_count += 1
                        print(f" ID {id} : correct | Step6_label: {step6_label} | label: {label}")
                    else:
                        f_mismatch.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                        wrong_count += 1
                        print(f" ID {id} : incorrect | Step6_label: {step6_label} | label: {label}")

                except json.JSONDecodeError as e:
                    error_count += 1
                    error_id.append(id)
                    print(f" ID {id} : JSON decode error - {str(e)}")

                except KeyError as e:
                    error_count += 1
                    error_id.append(id)
                    print(f" ID {id} : missing field - {str(e)}")

                except Exception as e:
                    error_count += 1
                    print(f" ID {id} : unknown error - {str(e)}")

    # Final summary
    print("\n" + "="*50)
    print("Processing completed. Summary:")
    print(f"Total samples: {total_count}")
    print(f"Matched: {correct_count} (saved to {correct_file})")
    print(f"Mismatched: {wrong_count} (saved to {wrong_file})")
    print(f"Errors: {error_count}, IDs: {error_id}")
    print(f"IDs with empty step labels: {empty_label_id}")
    


if __name__ == "__main__":

    input_path = "./univariate_0_2000_filtered_labeled_cot.jsonl"
    correct_path = "./univariate_0_2000_filtered_labeled_cot_stepLabeled_correct_test.jsonl"       # correct matches
    wrong_path = "./univariate_0_2000_filtered_labeled_cot_stepLabeled_wrong_test.jsonl"          # mismatches
    
    # Clear output files
    open(correct_path, 'w').close()
    open(wrong_path, 'w').close()

    # Run processing
    process_jsonl(input_path, correct_path, wrong_path)
