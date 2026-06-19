# lm-evaluation-harness 命令示例

## 安装

```bash
# 从 PyPI 安装
pip install lm-eval

# 或从源码安装
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

## 基础评估

### 在单个任务上评估 HuggingFace 模型
```bash
lm_eval --model hf \
    --model_args pretrained=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

### 在多个任务上评估
```bash
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-0.5B \
    --tasks hellaswag,arc_easy,winogrande \
    --device cuda:0 \
    --batch_size 16
```

### 使用少样本评估
```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks mmlu \
    --num_fewshot 5 \
    --device cuda:0 \
    --batch_size 4
```

## 高级用法

### 评估微调后的 LoRA 模型
```bash
# 先合并 LoRA 权重，然后评估
lm_eval --model hf \
    --model_args pretrained=my_finetuned_model,peft=my_lora_adapter \
    --tasks mmlu,hellaswag \
    --device cuda:0
```

### 将结果保存到指定目录
```bash
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME \
    --tasks TASK_NAME \
    --output_path ./results/my_experiment/ \
    --device cuda:0
```

### 使用特定精度评估
```bash
# FP16
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME,dtype=float16 \
    --tasks TASK_NAME \
    --device cuda:0

# BF16
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME,dtype=bfloat16 \
    --tasks TASK_NAME \
    --device cuda:0
```

### 多 GPU 评估
```bash
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME,parallelize=True \
    --tasks TASK_NAME \
    --batch_size auto
```

## 特定任务示例

### MMLU（大规模多任务语言理解）
```bash
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME \
    --tasks mmlu \
    --num_fewshot 5 \
    --device cuda:0
```

### HellaSwag（常识推理）
```bash
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME \
    --tasks hellaswag \
    --num_fewshot 0 \
    --device cuda:0
```

### ARC（科学问答）
```bash
# ARC-Easy
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME \
    --tasks arc_easy \
    --device cuda:0

# ARC-Challenge
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME \
    --tasks arc_challenge \
    --device cuda:0
```

### GSM8K（数学推理）
```bash
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME \
    --tasks gsm8k \
    --num_fewshot 5 \
    --device cuda:0
```

### TruthfulQA（真实性）
```bash
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME \
    --tasks truthfulqa \
    --device cuda:0
```

### HumanEval（代码生成）
```bash
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME \
    --tasks humaneval \
    --device cuda:0
```

## 完整排行榜评估

复现 Open LLM Leaderboard 评估：
```bash
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME \
    --tasks mmlu,hellaswag,arc_challenge,winogrande,truthfulqa,gsm8k \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size auto \
    --output_path ./results/leaderboard_eval/
```

## 列出可用任务

```bash
# 列出所有可用任务
lm_eval --tasks list

# 列出匹配模式的任务
lm_eval --tasks list --task_pattern mmlu*
```

## 比较两个模型

```bash
# 评估基础模型
lm_eval --model hf \
    --model_args pretrained=base_model \
    --tasks mmlu,hellaswag \
    --output_path results/base.json

# 评估微调后的模型
lm_eval --model hf \
    --model_args pretrained=finetuned_model \
    --tasks mmlu,hellaswag \
    --output_path results/finetuned.json

# 比较结果（手动比较 JSON 文件）
```
