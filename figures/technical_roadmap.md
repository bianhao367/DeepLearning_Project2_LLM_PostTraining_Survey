# 技术路线图

## LLM Post-training 技术路线图

使用 Mermaid 语法绘制，可粘贴到 https://mermaid.live 导出为 PNG/SVG/PDF。

```mermaid
flowchart TD
    A["预训练模型\n(Pretrained Model)\nLLaMA / Qwen / TinyLlama"] --> B["第一阶段：微调\n(Supervised Fine-tuning)"]

    subgraph B["第一阶段：微调"]
        B1["准备 Alpaca 风格数据\n(instruction / input / output)"]
        B2["选择微调方法"]
        B1 --> B2
        B2 --> B3a["全量微调\n(Full Fine-tuning)"]
        B2 --> B3b["LoRA\n(低秩适配)"]
        B2 --> B3c["QLoRA\n(量化 LoRA)"]
        B3a --> B4["监督微调模型 (SFT Model)"]
        B3b --> B4
        B3c --> B4
    end

    B4 --> C["第二阶段：偏好数据收集"]

    subgraph C["第二阶段：偏好数据收集"]
        C1["人工标注\n(Human Ranking)"]
        C2["AI 反馈\n(RLAIF / GPT-4)"]
        C1 --> C3["偏好对数据\n(chosen / rejected)"]
        C2 --> C3
    end

    C3 --> D["第三阶段：对齐\n(Alignment)"]

    subgraph D["第三阶段：对齐"]
        D1["SFT → 奖励模型 → RLHF/PPO"]
        D2["SFT → 偏好数据 → DPO"]
        D3["SFT → 推理 RL → GRPO"]
        D1 --> D4["对齐后的模型\n(Aligned Model)"]
        D2 --> D4
        D3 --> D4
    end

    D4 --> E["第四阶段：评测\n(Evaluation Harness)"]

    subgraph E["第四阶段：评测"]
        E1["lm-evaluation-harness\n(EleutherAI)"]
        E2["OpenCompass\n(上海AI实验室)"]
        E3["HELM\n(Stanford CRFM)"]
        E1 --> E4["基准测试结果\n(MMLU / HellaSwag\nARC / GSM8K\nHumanEval)"]
        E2 --> E4
        E3 --> E4
    end

    E4 --> F["错误分析与模型迭代\n(Error Analysis & Iteration)"]
```

## 如何使用

1. 复制上方 Mermaid 代码
2. 打开 https://mermaid.live
3. 粘贴代码，右侧即可预览
4. 点击导出按钮，选择 PNG 或 SVG 格式下载
5. 将下载的图片命名为 `technical_roadmap.png` 放入 `figures/` 目录

## 替代方案

也可以使用以下工具绘制：
- **draw.io**：https://app.diagrams.net（免费在线工具）
- **PowerPoint**：使用流程图模板
- **Visio**：专业流程图工具
- **LaTeX TikZ**：学术论文制图
