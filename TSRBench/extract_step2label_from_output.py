import json
import re
import time
from openai import OpenAI

# Configure OpenAI client
gpt_model = "gpt-4o-mini"
OPENAI_API_KEY = ""  # Replace with your API key
client = OpenAI(api_key=OPENAI_API_KEY, base_url="")

# Function for LLM API requests
def gpt_chat(content, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                temperature=0.2,
                messages=[{"role": "user", "content": content}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API request failed (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(5)
    print("Maximum number of retries reached. Request failed.")
    return None

prompt_template = """
Please complete the task based on the following instructions:
1.First, extract time-series key patterns from the explanatory part of the question answer, which contains both a conclusion and supporting explanations. The extraction must comply with two rules:
- **Rule 1**: Patterns must be actionable and critical to deriving valid conclusions, including temporal patterns (e.g., trend; amplitude; fluctuation; continuity), judgment criteria (e.g., task-specific definitions of patterns), threshold values (e.g., upper bounds; lower bounds; percentage deviations), and other decisive patterns or criteria.
- **Rule 2**: The extracted patterns must be specific, concrete time-series pattern names (e.g., trend; amplitude; fluctuation; continuity) that directly describe the time-series characteristic. Do NOT include specific numerical values, quantitative details, or descriptive modifiers.
2.Compare the extracted key patterns (following Rule 1 and Rule 2) with the provided patterns. If the extracted key patterns are not included in the provided patterns, supplement these patterns to the provided patterns.
Given Information:
Answer: {output}
Patterns: {step2_label}
Output Format:
<Only list the complete key pattern names after supplementation (including the patterns in the original and the newly added patterns), no extra details, analysis or conclusions; separate multiple items with semicolons>.
"""

def process_jsonl_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            
            # Extract required fields
            output = data.get('output', 'unknown')
            id = data.get('id', 'unknown')
            original_label = data.get('step2_label', 'unknown')
           
            prompt = prompt_template.format(output=output, step2_label=original_label)
            updated_label = gpt_chat(prompt)
            
            data['step2_label'] = updated_label if updated_label else "unknown"

            # Write to output file
            outfile.write(json.dumps(data) + '\n')
            print(f"ID {id}: step2_label updated")
            print(f"BEFORE: {original_label}")
            print(f"AFTER : {updated_label}\n")
            
            time.sleep(1)


if __name__ == "__main__":
    input_filename = "./univariate_0_2000_filtered_labeled_cot_stepLabeled_correct.jsonl"
    output_filename = "./univariate_0_2000_filtered_labeled_cot_stepLabeled_correct_step2label.jsonl"
    
    # Clear output file
    open(output_filename, 'w').close()
    
    process_jsonl_file(input_filename, output_filename)
