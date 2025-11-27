# Pipeline for Curating TSRBench

The **Scenario-Based** Datasets of TSRBench includes three reasoning tasks: **Anomaly Detection**, **Scenario Attribution**, and **Inferential Calculation**.

The dataset is curated based on the [ChatTS Training Dataset](https://huggingface.co/datasets/ChatTSRepo/ChatTS-Training-Dataset/tree/main)

### 1. Task Categorization

A rule-based extractor is applied to select samples that meet predefined reasoning criteria.

Task-specific keyword sets are designed to automate task assignment for all three reasoning types.

🔧 **Script**: `classify_rule_based.py`

✅ **Quality Control**: ~5% of samples are manually inspected to verify classification accuracy.

### 2. Label Extraction

Ground-truth labels are extracted using regular-expression–based pattern matching.

🔧 **Script**: `extract_label.py`

### 3. TS-Tailored Thinking Strategy

DeepSeek-R1 is used to generate structured multi-step reasoning traces. All tasks follow a unified 6-step reasoning template:

**Task Intent Identification -> Candidate Attributes Selection -> Critical Segments Analysis  -> Preliminary Answer Formulation -> Backtracking and Self-Reflection -> Reasoning Process Summarization**


🔧 **Script**: `cot_deepseekr1.py` (contains reasoning templates for all three tasks)

### 4. Output Validation & Verifiable Step-Label Extraction

DeepSeek-R1 outputs are validated for correctness, while verifiable step-level supervision signals are extracted simultaneously.

🔧 **Script**: `cot_correct.py`

### 5. Final CoT Generation

The final CoT path in TSRBench is standardized into the following format:

```
<THINK>{cot_deepseekr1}</THINK><ANSWER>The answer is {label}.</ANSWER>
```

🔧 **Script**: `generate_cot.py`

### 6. Step 2 Label Enhancement

Task-relevant temporal patterns from the original ChatTS dataset are integrated to refine the Step 2: Candidate Attribute Selection labels.

🔧 **Script**: `extract_step2label_from_output.py`

### 7. Dataset Statistics

Scripts for computing dataset statistics

🔧 **Script**: `classify_cnt.py`
