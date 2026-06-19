# 项目 2 任务 B：大语言模型后训练综述

## 概述

本项目是对大语言模型（LLM）后训练技术的综合综述，涵盖三个关键领域：

1. **微调（Fine-tuning）**：将预训练模型适配到特定任务
2. **对齐（Alignment）**：确保模型遵循人类价值观
3. **评测框架（Evaluation Harness）**：评估模型能力的标准化框架

## 项目结构

```
Project2_TaskB_Survey/
│
├── README.md                           # 本文件
│
├── report/
│   └── project_report.md               # 完整综述报告
│
├── figures/
│   ├── post_training_pipeline.md       # 流程图（ASCII art）
│   ├── method_comparison_table.md      # 方法对比表
│   └── technical_roadmap.md            # 技术路线图（Mermaid）
│
├── notes/
│   ├── finetuning_notes.md             # 第一部分：微调笔记
│   ├── alignment_notes.md              # 第二部分：对齐笔记
│   └── evaluation_harness_notes.md     # 第三部分：评测框架笔记
│
├── examples/
│   ├── lm_eval_example_commands.md     # lm-evaluation-harness 命令示例
│   └── tool_usage_examples.md          # 其他工具使用示例
│
└── references/
    └── references.md                   # 完整参考文献
```

## 内容摘要

### 第一部分：微调
- 微调概念与动机
- 预训练 vs 微调 vs 指令微调
- 全量微调 vs 参数高效微调
- LoRA（低秩适配）原理与实现
- QLoRA（量化 LoRA）实现显存高效训练
- Stanford Alpaca 案例研究
- 方法综合对比

### 第二部分：对齐
- 对齐概念与目标（有帮助、无害、诚实）
- RLHF（基于人类反馈的强化学习）流程
- 奖励模型训练
- PPO（近端策略优化）在 RLHF 中的应用
- DPO（直接偏好优化）
- RLAIF（基于 AI 反馈的强化学习）
- GRPO（群组相对策略优化）
- 对齐与推理能力

### 第三部分：评测框架
- 评测框架概念
- 常见基准测试（MMLU、HellaSwag、ARC、GSM8K、HumanEval）
- EleutherAI lm-evaluation-harness
- Hugging Face 评测工具
- OpenCompass 评测平台
- HELM（语言模型整体评估）
- 评测最佳实践

## 核心参考文献

| 论文 | 年份 | 主题 |
|------|------|------|
| LoRA (Hu et al.) | 2021 | 参数高效微调 |
| QLoRA (Dettmers et al.) | 2023 | 量化 LoRA |
| InstructGPT (Ouyang et al.) | 2022 | RLHF 基础 |
| DPO (Rafailov et al.) | 2023 | 直接偏好优化 |
| GRPO (Shao et al.) | 2024 | 群组相对策略优化 |
| HELM (Liang et al.) | 2022 | 整体评估框架 |

完整参考文献请参见 `references/references.md`。

## 使用方法

1. **阅读报告**：从 `report/project_report.md` 开始了解完整综述
2. **查阅笔记**：使用 `notes/` 获取详细的专题信息
3. **运行示例**：运行 `examples/` 中的命令实践评测工具
4. **查看图表**：查看 `figures/` 获取可视化摘要和对比
5. **引用文献**：使用 `references/references.md` 进行引用

## 技术路线图

完整的 LLM 后训练流程如下：

```
预训练模型 → 微调（全量/LoRA/QLoRA）→ 偏好数据收集 → 对齐（RLHF/DPO/GRPO）→ 评测 → 错误分析
```

详细技术路线图请参见 `figures/technical_roadmap.md`（Mermaid 格式）。

## 评测框架使用说明

### lm-evaluation-harness（EleutherAI）

```bash
# 安装
pip install lm-eval

# 基础评测
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-0.5B \
    --tasks hellaswag,mmlu \
    --device cuda:0 \
    --batch_size 8

# 少样本评测
lm_eval --model hf \
    --model_args pretrained=MODEL_NAME \
    --tasks mmlu \
    --num_fewshot 5 \
    --device cuda:0
```

### OpenCompass（上海人工智能实验室）

```bash
pip install opencompass
python run.py --models hf_llama2_7b --datasets mmlu_ppl
```

### HELM（Stanford CRFM）

```bash
pip install crfm-helm
helm-run --run-entries mmlu:model=openai/gpt-3.5-turbo --suite my-suite
helm-summarize --suite my-suite
```

更多示例请参见 `examples/lm_eval_example_commands.md` 和 `examples/tool_usage_examples.md`。

## LLM 后训练总结

本综述涵盖了大语言模型后训练的三大支柱：

1. **微调**：将预训练模型适配到特定任务。LoRA 和 QLoRA 以极小的显存成本实现了接近全量微调的性能，使得在消费级 GPU 上微调大模型成为可能。
2. **对齐**：确保模型按照人类价值观行事。RLHF 是主流范式，DPO 提供了更简单的替代方案，GRPO 在推理任务上表现出色。
3. **评测框架**：为衡量模型能力提供了标准化系统。lm-evaluation-harness 是最广泛使用的开源框架，OpenCompass 在中文评测方面表现突出。

该领域正在快速发展，新方法不断涌现。随着 LLM 能力的持续增长，稳健的后训练和评测对于确保其安全部署将变得越来越重要。

## 工具与框架

### 微调相关
- Hugging Face Transformers
- Hugging Face PEFT（LoRA/QLoRA）
- Hugging Face TRL（RLHF/DPO）
- bitsandbytes（量化）

### 评测相关
- EleutherAI lm-evaluation-harness
- OpenCompass
- HELM
- Hugging Face evaluate

## 课程信息

- 课程：深度学习
- 项目：项目 2 - 大语言模型后训练
- 任务：任务 B - 微调、对齐与评测框架综述

## 小组成员

- 程曦
