import re
import json
from typing import List, Dict
from word2number import w2n 

'''
Manual verification: ensure stepx labels are non-empty and correct before generating the final COT.
'''

def generate_cot_field(cot_content: str, step6_label: str | None) -> str:
    cot_clean = cot_content.strip() if isinstance(cot_content, str) else ""
    
    # Remove trailing punctuation
    if step6_label and isinstance(step6_label, str):
        step6_clean = re.sub(r'[.;]$', '', step6_label).strip()
    else:
        step6_clean = "unknown"

    answer_part = f"The answer is {step6_clean}." if step6_clean else "The answer is unknown."
    return f"<think>{cot_clean}</think><ANSWER>{answer_part}</ANSWER>"


def process_jsonl(input_file: str, output_file: str) -> None:
    error_count = 0
    error_id = []
    
    with open(output_file, 'w', encoding="utf-8") as f_out:

        with open(input_file, 'r', encoding="utf-8") as f_in:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Validate required fields
                    required_fields = ["id", "cot_deepseekr1", "step6_label"]
                    for field in required_fields:
                        if field not in data:
                            raise KeyError(f"Missing required field: {field}")
                        
                    id = data["id"]
                    step6_label = data.get("step6_label") or "unknown"
                    cot_deepseekr1 = data["cot_deepseekr1"]
                    
                    cot_field = generate_cot_field(cot_deepseekr1, step6_label)

                    # Build new dict and insert 'cot' after 'label' (if exists)
                    new_data = {}
                    for key, value in data.items():
                        new_data[key] = value
                        if key == "label":
                            new_data['cot'] = cot_field
                            
                    json.dump(new_data, f_out, ensure_ascii=False, indent=None)
                    f_out.write('\n')

                    print(f" ID {id} : processed successfully | step6_label: {step6_label}")

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


if __name__ == "__main__":

    input_path = "./univariate_0_2000_filtered_labeled_cot_stepLabeled_correct_test.jsonl"
    output_path = "./univariate_0_2000_filtered_labeled_cot_stepLabeled_correct_test2.jsonl" 
    
    # Clear output file
    open(output_path, 'w').close()

    # Run batch processing
    process_jsonl(input_path, output_path)
