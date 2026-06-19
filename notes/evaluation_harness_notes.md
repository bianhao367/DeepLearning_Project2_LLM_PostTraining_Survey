# 评测框架笔记

## 1. 什么是评测框架？

评测框架是一个标准化系统，用于在多个基准测试、任务和指标上系统地评估语言模型。它提供：

- **一致性**：跨模型使用相同的评估协议
- **可复现性**：标准化的提示、指标和设置
- **可比性**：模型之间的公平比较
- **自动化**：跨多任务的批量评估

### 为什么需要评测框架
- 人工评估成本高且不一致
- 不同论文使用不同的评估设置，难以比较
- 需要标准化基准测试来跟踪领域进展

## 2. 核心概念

### 基准测试（Benchmark）
设计用于测试特定模型能力的任务集合：
- **MMLU**：多任务语言理解（57 个学科）
- **HellaSwag**：常识推理
- **ARC**：科学问答（Easy/Challenge）
- **GSM8K**：小学数学
- **HumanEval**：代码生成
- **TruthfulQA**：真实性
- **WinoGrande**：共指消解

### 评估指标
不同任务使用不同指标：
- **准确率（Accuracy）**：精确匹配准确率（分类任务）
- **困惑度（Perplexity）**：语言建模质量
- **Pass@k**：k 次尝试中生成正确代码的概率
- **BLEU/ROUGE**：文本生成质量
- **F1**：Token 级别 F1 分数（QA 任务）

### 少样本设置
- **零样本（0-shot）**：提示中无示例
- **少样本（k-shot）**：提示中提供 k 个示例（通常 1-5 个）
- 少样本通常通过提供任务上下文来提升性能
- 标准做法：使用 0-shot 和 5-shot 进行评估

## 3. EleutherAI lm-evaluation-harness

### 概述
最广泛使用的 LLM 开源评测框架。由 EleutherAI 开发。

### 架构
```
lm-evaluation-harness/
├── lm_eval/
│   ├── api/          # 模型和任务接口
│   ├── evaluator/    # 核心评估逻辑
│   ├── models/       # 模型实现（HF、vLLM 等）
│   └── tasks/        # 任务定义
└── examples/         # 使用示例
```

### 主要特点
- 200+ 任务和基准测试
- 支持多种模型后端（HuggingFace、vLLM、OpenAI API 等）
- 可配置的少样本设置
- 标准化的指标计算
- 多 GPU 评估支持

### 基本用法
```bash
# 安装
pip install lm-eval

# 评估 HuggingFace 模型
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks hellaswag,mmlu \
    --device cuda:0 \
    --batch_size 8

# 指定少样本数量评估
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B \
    --tasks mmlu \
    --num_fewshot 5 \
    --device cuda:0

# 评估多个任务
lm_eval --model hf \
    --model_args pretrained=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tasks arc_easy,arc_challenge,winogrande,truthfulqa \
    --device cuda:0 \
    --output_path results/
```

### 输出格式
结果以 JSON 格式保存：
```json
{
  "results": {
    "hellaswag": {
      "acc": 0.5523,
      "acc_norm": 0.7312,
      "acc_stderr": 0.0050
    }
  },
  "config": {
    "model": "hf",
    "model_args": "pretrained=...",
    "num_fewshot": 0
  }
}
```

## 4. Hugging Face 评测工具

### evaluate 库
```python
from evaluate import load

# 加载指标
accuracy = load("accuracy")

# 计算准确率
results = accuracy.compute(references=[0, 1, 2], predictions=[0, 1, 1])
print(results)  # {'accuracy': 0.6667}
```

### Open LLM Leaderboard
- Hugging Face 托管的开源 LLM 排行榜
- 底层使用 EleutherAI lm-evaluation-harness
- 当前基准测试：MMLU、HellaSwag、ARC、GSM8K、TruthfulQA、WinoGrande
- 网址：https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard

### AutoEval
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate

model = AutoModelForCausalLM.from_pretrained("model_name")
tokenizer = AutoTokenizer.from_pretrained("model_name")

results = simple_evaluate(
    model="hf",
    model_args="pretrained=model_name",
    tasks=["mmlu", "hellaswag"],
    batch_size=8
)
```

## 5. OpenCompass

### 概述
OpenCompass 是由上海人工智能实验室开发的综合评测平台。它提供：
- 100+ 数据集覆盖各种能力
- 模块化评测架构
- 支持中英文基准测试
- 分布式评测支持

### 主要特点
- **多维度评估**：语言、知识、推理、代码等
- **中文基准测试**：C-Eval、CMMLU、GAOKAO 等
- **配置系统**：灵活的任务配置
- **排行榜**：OpenCompass 排名

### 用法
```bash
# 安装
pip install opencompass

# 运行评估
python run.py --models hf_llama2_7b --datasets mmlu_ppl

# 使用配置文件运行
python run.py configs/eval_demo.py
```

### 配置示例
```python
from mmengine.config import read_base

with read_base():
    from .datasets.mmlu.mmlu_ppl import mmlu_datasets
    from .models.hf_llama2.hf_llama2_7b import models

datasets = mmlu_datasets
models = models
```

## 6. HELM（语言模型整体评估）

### 概述
由 Stanford CRFM 开发。HELM 提供全面的多维度评估框架。

### 主要特点
- **整体性**：在多个维度上评估（准确率、校准、鲁棒性、公平性、偏见、毒性、效率）
- **7 个场景**：覆盖不同能力的核心场景
- **共 42 个场景**：全面覆盖
- **标准化**：固定的提示、指标和评估协议

### 评估场景
1. **语言建模**：原始语言建模能力
2. **问答**：阅读理解、开放式问答
3. **信息检索**：文档检索和排序
4. **摘要**：文本摘要
5. **情感分析**：二元情感分类
6. **毒性检测**：识别有害内容
7. **偏见与公平性**：衡量人口统计学偏见

### 用法
```bash
# 安装
pip install crfm-helm

# 运行评估
helm-run --run-entries mmlu:model=openai/gpt-3.5-turbo --suite my-suite

# 汇总结果
helm-summarize --suite my-suite
```

## 7. 评测最佳实践

### 1. 选择合适的基准测试
- 匹配基准测试与你的使用场景
- 不要只挑选有利的基准测试
- 包含简单和困难的任务

### 2. 使用一致的设置
- 相同的少样本数量
- 相同的提示模板
- 相同的批大小和精度

### 3. 报告置信区间
- 始终报告标准误差
- 使用足够样本以获得统计显著性
- 尽可能使用 bootstrap 置信区间

### 4. 注意数据污染
- 训练数据可能与基准测试数据重叠
- 报告污染检测方法
- 尽可能使用留出的基准测试

### 5. 公平比较
- 所有模型使用相同的评估设置
- 相同的硬件和精度
- 报告所有指标，而非仅有利的

## 8. 快速参考：常用命令

### lm-evaluation-harness
```bash
# 单任务
lm_eval --model hf --model_args pretrained=MODEL --tasks TASK --device cuda:0

# 多任务
lm_eval --model hf --model_args pretrained=MODEL --tasks task1,task2,task3

# 少样本
lm_eval --model hf --model_args pretrained=MODEL --tasks TASK --num_fewshot 5

# 保存结果
lm_eval --model hf --model_args pretrained=MODEL --tasks TASK --output_path results/
```

### OpenCompass
```bash
# 基础评估
python run.py --models MODEL --datasets DATASET

# 使用配置
python run.py configs/my_config.py

# 列出可用任务
python tools/list_configs.py
```

### HELM
```bash
# 运行特定场景
helm-run --run-entries SCENARIO:MODEL --suite SUITE_NAME

# 汇总
helm-summarize --suite SUITE_NAME
```
