# VeriTime: Enhancing LLM Time Series Reasoning via Verifiable CoT Path and Reinforcement Learning

## TSRBench

We open-source the TSRBench dataset on **Dataset** folder.

The **TSRBench** folder contains the curation pipeline for our dataset.
TSRBench consists of two major categories of reasoning tasks: **Scenario-based** and **Knowledge-based**, with 7 subsets in total.

#### Scenario-based Reasoning
| Reasoning Type | Task |
|----------------|------|
| Deductive Reasoning | Anomaly Detection |
| Causal Reasoning | Scenario Attribution |
| Quantitative Reasoning | Inferential Calculation |

#### Knowledge-based Reasoning  
| Reasoning Type | Task |
|----------------|------|
| Inductive Reasoning | CTU (Computer Type Usage) |
| | ECG (Electrocardiogram Record Diagnosis) |
| | EMG (Electromyogram Signal Diagnosis) |
| | RCW (Right Whale Calls Detection) |


## Experiments

#### Environment Setup
Our environment is based on **Python 3.12**.

To install all dependencies:

```bash
pip install -r requirements.txt
```

#### Model Download

We use Qwen2.5-3B-Instruct and Qwen3-4B-Instruct as the base models.
The models can be downloaded from the [official Qwen Hugging Face collection](https://huggingface.co/Qwen/collections).

### VeriTime
#### Supervised Fine-Tuning (SFT)
In the first stage, we fine-tune the model to improve stability and strengthen general reasoning capability:

```bash
python VeriTime_sft.py
```
#### Reinforcement Learning (GRPO)
To further enhance step-by-step reasoning, we perform reinforcement learning using the GRPO algorithm.

This stage leverages our fine-grained, multi-objective reward functions designed to evaluate intermediate reasoning steps:

```bash
python VeriTime_grpo.py
```

#### Evaluation

After the two-stage RFT training, evaluate the final model on the test set using:

```bash
python evaluate.py
```

#### RFT data selection

In our RFT data selection process, we categorize sample difficulty based on  task taxonomy defined. A warm-up model is then used to distinguish between **normally learnable samples** and **promising challenging samples**.

We additionally provide several baseline selection strategies in the directory.

```bash
python ifd_select.py # Instruction-Following Difficulty
python ppl_select.py # Perplexity
python length_select.py # Token Length
```

 #### Result

We release the test results of Qwen models trained under VeriTime in the **LLM_output** folder.