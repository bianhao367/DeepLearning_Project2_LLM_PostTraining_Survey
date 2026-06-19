# 参考文献

## 微调

1. **LoRA: Low-Rank Adaptation of Large Language Models**
   - Hu, E. J., et al. (2021)
   - arXiv:2106.09685
   - 提出 LoRA 用于参数高效微调

2. **QLoRA: Efficient Finetuning of Quantized LLMs**
   - Dettmers, T., et al. (2023)
   - arXiv:2305.14314
   - 4 位量化 + LoRA 实现显存高效微调

3. **Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning**
   - Lialin, V., et al. (2023)
   - arXiv:2303.15647
   - PEFT 方法综合综述

4. **Prefix-Tuning: Optimizing Continuous Prompts for Generation**
   - Li, X. L., & Liang, P. (2021)
   - arXiv:2101.00190
   - 连续提示微调方法

5. **AdapterHub: A Framework for Adapting Transformers**
   - Pfeiffer, J., et al. (2020)
   - arXiv:2007.07779
   - 基于适配器的微调框架

## 指令微调

6. **Stanford Alpaca: An Instruction-following LLaMA model**
   - Taori, R., et al. (2023)
   - GitHub: tatsu-lab/stanford_alpaca
   - 展示了低成本的指令微调方法

7. **Self-Instruct: Aligning Language Models with Self-Generated Instructions**
   - Wang, Y., et al. (2023)
   - arXiv:2212.10560
   - 自动生成指令数据的方法

8. **Scaling Instruction-Finetuned Language Models (Flan-T5/PaLM)**
   - Chung, H. W., et al. (2022)
   - arXiv:2210.11416
   - 指令微调的规模化

9. **LIMA: Less Is More for Alignment**
   - Zhou, C., et al. (2023)
   - arXiv:2305.11206
   - 指令微调中数据质量优于数量

## 对齐

10. **Training language models to follow instructions with human feedback (InstructGPT)**
    - Ouyang, L., et al. (2022)
    - arXiv:2203.02155
    - RLHF 的奠基性论文

11. **Learning to summarize from human feedback**
    - Stiennon, N., et al. (2020)
    - arXiv:2009.01325
    - 摘要领域的早期 RLHF 工作

12. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**
    - Rafailov, R., et al. (2023)
    - arXiv:2305.18290
    - DPO：无需 RL 的简化对齐方法

13. **Constitutional AI: Harmlessness from AI Feedback**
    - Bai, Y., et al. (2022)
    - arXiv:2212.08073
    - RLAIF 和 Constitutional AI 方法

14. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO)**
    - Shao, Z., et al. (2024)
    - arXiv:2402.03300
    - 群组相对策略优化

15. **Proximal Policy Optimization Algorithms**
    - Schulman, J., et al. (2017)
    - arXiv:1707.06347
    - RLHF 中使用的 PPO 算法

16. **Reward Modeling**
    - Christiano, P. F., et al. (2017)
    - "Deep reinforcement learning from human preferences"
    - arXiv:1706.03741

## 评测

17. **A Language Model for the 21st Century (HELM)**
    - Liang, P., et al. (2022)
    - "Holistic Evaluation of Language Models"
    - arXiv:2211.09110

18. **Measuring Massive Multitask Language Understanding (MMLU)**
    - Hendrycks, D., et al. (2021)
    - arXiv:2009.03300
    - 57 任务的 LLM 评估基准

19. **HellaSwag: Can a Machine Really Finish Your Sentence?**
    - Zellers, R., et al. (2019)
    - arXiv:1905.07830
    - 常识推理基准测试

20. **Think you have Solved Question Answering? Try ARC**
    - Clark, P., et al. (2018)
    - arXiv:1803.05457
    - AI2 推理挑战

21. **Training Verifiers to Solve Math Word Problems (GSM8K)**
    - Cobbe, K., et al. (2021)
    - arXiv:2110.14168
    - 小学数学基准测试

22. **Evaluating Large Language Models Trained on Code (HumanEval)**
    - Chen, M., et al. (2021)
    - arXiv:2107.03374
    - 代码生成基准测试

23. **Measuring Massive Multitask Language Understanding in Chinese (C-Eval)**
    - Huang, Y., et al. (2023)
    - arXiv:2305.08322
    - 中文语言基准测试

24. **CMMLU: Measuring Chinese Massive Multitask Language Understanding**
    - Li, H., et al. (2023)
    - arXiv:2306.09212
    - 中文基准测试

25. **OpenCompass: A Universal Evaluation Platform for Foundation Models**
    - Contributors (2023)
    - GitHub: open-compass/opencompass
    - 综合评测平台

26. **lm-evaluation-harness**
    - Gao, L., et al. (2023)
    - GitHub: EleutherAI/lm-evaluation-harness
    - LLM 标准评测框架

## 通用 LLM 参考

27. **Attention Is All You Need**
    - Vaswani, A., et al. (2017)
    - arXiv:1706.03762
    - Transformer 架构

28. **BERT: Pre-training of Deep Bidirectional Transformers**
    - Devlin, J., et al. (2019)
    - arXiv:1810.04805
    - 预训练范式

29. **Language Models are Few-Shot Learners (GPT-3)**
    - Brown, T., et al. (2020)
    - arXiv:2005.14165
    - 规模化与上下文学习

30. **LLaMA: Open and Efficient Foundation Language Models**
    - Touvron, H., et al. (2023)
    - arXiv:2302.13971
    - 开源 LLM 基础模型

31. **Qwen2.5 Technical Report**
    - Qwen Team (2024)
    - Alibaba Cloud
    - Qwen2.5 模型系列

32. **TinyLlama: An Open-Source Small Language Model**
    - Zhang, P., et al. (2024)
    - arXiv:2401.02385
    - 11 亿参数语言模型
