# 工具使用示例

## 1. Hugging Face PEFT（LoRA 微调）

### 安装
```bash
pip install transformers peft accelerate bitsandbytes
```

### LoRA 微调示例
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset

# 加载模型和分词器
model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # 秩
    lora_alpha=32,                 # 缩放因子
    lora_dropout=0.05,             # Dropout
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 应用到注意力层
)

# 将 LoRA 应用到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出：trainable params: 1,234,567 || all params: 500,000,000 || trainable%: 0.247%

# 加载数据集
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 格式化数据
def format_prompt(example):
    if example["input"]:
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

# 训练参数
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# 训练
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=format_prompt,
)
trainer.train()

# 保存 LoRA 适配器
model.save_pretrained("./lora_adapter")
```

### QLoRA 微调示例
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4 位量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,       # 双重量化
)

# 使用 4 位量化加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    quantization_config=bnb_config,
    device_map="auto",
)

# 应用 LoRA（同上）
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05)
model = get_peft_model(model, lora_config)
```

## 2. Hugging Face TRL（RLHF/DPO）

### DPO 训练示例
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# 加载偏好数据集
# 每个样本应包含：prompt, chosen, rejected
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# DPO 配置
dpo_config = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-7,
    beta=0.1,                    # KL 惩罚系数
    logging_steps=10,
)

# 训练
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### PPO 训练示例（RLHF）
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 加载带价值头的模型
model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# PPO 配置
ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
)

# 初始化 PPO 训练器
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

# 训练循环（简化版）
for batch in dataloader:
    query_tensors = [tokenizer.encode(q, return_tensors="pt") for q in batch["query"]]
    
    # 生成响应
    response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=128)
    
    # 从奖励模型获取奖励
    rewards = [reward_model(q, r) for q, r in zip(query_tensors, response_tensors)]
    
    # PPO 步骤
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

## 3. Hugging Face evaluate 库

### 基本用法
```python
from evaluate import load

# 加载指标
accuracy = load("accuracy")
f1 = load("f1")
bleu = load("bleu")
rouge = load("rouge")

# 计算准确率
results = accuracy.compute(references=[0, 1, 2, 3], predictions=[0, 1, 1, 3])
print(f"准确率: {results['accuracy']}")

# 计算 F1
results = f1.compute(references=[0, 1, 2], predictions=[0, 1, 1], average="macro")
print(f"F1: {results['f1']}")

# 计算 BLEU
results = bleu.compute(
    predictions=["猫坐在垫子上"],
    references=[["猫在垫子上"]]
)
print(f"BLEU: {results['bleu']}")
```

### 自定义评估脚本
```python
from evaluate import load
import numpy as np

def evaluate_model(model, tokenizer, dataset, task="accuracy"):
    metric = load(task)
    predictions = []
    references = []
    
    for example in dataset:
        # 生成预测
        inputs = tokenizer(example["question"], return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        predictions.append(pred)
        references.append(example["answer"])
    
    # 计算指标
    results = metric.compute(predictions=predictions, references=references)
    return results
```

## 4. OpenCompass 使用

### 安装
```bash
pip install opencompass
```

### 基础评估
```bash
# 列出可用数据集
python tools/list_configs.py

# 使用内置配置运行评估
python run.py --models hf_llama2_7b --datasets mmlu_ppl

# 使用自定义配置运行
python run.py configs/eval_my_model.py
```

### 自定义配置
```python
# configs/eval_my_model.py
from mmengine.config import read_base

with read_base():
    from .datasets.mmlu.mmlu_ppl_5shot import mmlu_datasets
    from .datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    from .models.hf_llama2.hf_llama2_7b import models

datasets = mmlu_datasets + hellaswag_datasets
models = models
```

## 5. HELM 使用

### 安装
```bash
pip install crfm-helm
```

### 基础评估
```bash
# 运行 MMLU 场景
helm-run --run-entries mmlu:model=openai/gpt-3.5-turbo --suite my-suite

# 运行多个场景
helm-run --run-entries "mmlu:model=openai/gpt-3.5-turbo,truthfulqa:model=openai/gpt-3.5-turbo" --suite my-suite

# 汇总结果
helm-summarize --suite my-suite

# 查看结果
# 结果保存在 benchmark_output/runs/my-suite/
```

## 6. 数据格式示例

### Alpaca 格式
```json
{
    "instruction": "给出保持健康的三个建议。",
    "input": "",
    "output": "1. 均衡饮食，多吃水果和蔬菜。2. 规律运动，每天至少 30 分钟。3. 保证 7-9 小时的优质睡眠。"
}
```

### DPO 偏好格式
```json
{
    "prompt": "用简单的话解释量子计算。",
    "chosen": "量子计算使用量子比特（qubit），它可以同时处于 0 和 1 的状态，使计算机能够比传统计算机更快地解决某些问题。",
    "rejected": "量子计算就是用量子的东西让计算机变快。非常复杂，涉及很多数学。"
}
```

### 对话格式（用于指令微调）
```json
{
    "messages": [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是人工智能的一个子领域，计算机从数据中学习模式而无需显式编程。"}
    ]
}
```
