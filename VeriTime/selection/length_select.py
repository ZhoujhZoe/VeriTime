import json
from transformers import AutoTokenizer
import os

def simple_split_by_cot_length():
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "PATH/TO/MODEL", 
        trust_remote_code=True
    )
    
    # 读取数据
    with open("PATH/TO/TRAIN_DATA.jsonl", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 计算每个case的cot字段token长度
    data_with_length = []
    for line in lines:
        data = json.loads(line.strip())
        cot_text = data.get("cot", "")
        tokens = tokenizer.encode(cot_text, add_special_tokens=False)
        data_with_length.append((line, len(tokens)))
    
    # 按token长度排序（从高到低）
    data_with_length.sort(key=lambda x: x[1], reverse=True)
    
    # 分割数据
    sft_data = [item[0] for item in data_with_length[:1539]]
    grpo_data = [item[0] for item in data_with_length[1539:]]
    
    # 保存文件
    os.makedirs("./length", exist_ok=True)
    
    with open("./length/chattsv1_sft.jsonl", 'w', encoding='utf-8') as f:
        f.writelines(sft_data)
    
    with open("./length/chattsv1_grpo.jsonl", 'w', encoding='utf-8') as f:
        f.writelines(grpo_data)
    
    print(f"SFT子集: {len(sft_data)} 条数据")
    print(f"GRPO子集: {len(grpo_data)} 条数据")

# 运行
simple_split_by_cot_length()