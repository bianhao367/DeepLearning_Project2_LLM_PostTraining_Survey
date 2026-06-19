# -*- coding: utf-8 -*-
"""
生成 PDF 报告和图片资源
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table
import matplotlib.font_manager as fm
import numpy as np
from fpdf import FPDF

# 设置中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一部分：生成图片
# ============================================================

def create_post_training_pipeline():
    """生成后训练流程图"""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    ax.set_title('LLM Post-training Pipeline\n后训练流程', fontsize=16, fontweight='bold', pad=20,
                 fontfamily='Microsoft YaHei')

    boxes = [
        (5, 18, 'Pretrained Model\n预训练模型\n(LLaMA, Qwen)', '#E8F5E9'),
        (5, 15, 'Stage 1: Fine-tuning\n阶段1: 微调\n(LoRA / QLoRA / Full FT)', '#BBDEFB'),
        (5, 12, 'Stage 2: Preference Data\n阶段2: 偏好数据收集\n(Human / AI Feedback)', '#FFF9C4'),
        (5, 9, 'Stage 3: Alignment\n阶段3: 对齐\n(RLHF/PPO / DPO / GRPO)', '#F8BBD0'),
        (5, 6, 'Stage 4: Evaluation\n阶段4: 评测\n(lm-eval / OpenCompass / HELM)', '#D1C4E9'),
        (5, 3, 'Results\n基准测试结果与错误分析\n(MMLU, HellaSwag, ARC, GSM8K)', '#C8E6C9'),
    ]

    for x, y, text, color in boxes:
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='#333333', linewidth=2)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontfamily='Microsoft YaHei',
                bbox=bbox)

    # Arrows
    for i in range(len(boxes) - 1):
        ax.annotate('', xy=(5, boxes[i+1][1] + 0.6), xytext=(5, boxes[i][1] - 0.6),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    plt.tight_layout()
    output_path = 'D:/python_code/transformer/Project2_TaskB_Survey/figures/post_training_pipeline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'已生成: {output_path}')


def create_method_comparison_table():
    """生成方法对比表"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    fig.suptitle('Method Comparison Tables\n方法对比表', fontsize=16, fontweight='bold',
                 fontfamily='Microsoft YaHei', y=0.98)

    # Table 1: Fine-tuning comparison
    ax1 = axes[0]
    ax1.axis('off')
    ax1.set_title('Fine-tuning Methods Comparison / 微调方法对比', fontsize=12, fontweight='bold',
                  fontfamily='Microsoft YaHei', pad=10)

    ft_data = [
        ['Aspect/方面', 'Full FT/全量微调', 'LoRA', 'QLoRA'],
        ['Trainable Params/可训练参数', '100%', '~0.1-1%', '~0.1-1%'],
        ['GPU Memory 7B/显存', '~60 GB', '~16 GB', '~6 GB'],
        ['GPU Memory 13B/显存', '~120 GB', '~30 GB', '~12 GB'],
        ['GPU Memory 65B/显存', '~500 GB', '~120 GB', '~48 GB'],
        ['Training Speed/训练速度', 'Baseline/基准', '1.2-1.5x faster', '0.8-1.0x'],
        ['Performance/性能', 'Best/最佳', '~98%', '~95%'],
        ['Min GPU/最低GPU', 'A100 80GB', 'RTX 3090 24GB', 'T4 16GB'],
    ]

    table1 = ax1.table(cellText=ft_data, loc='center', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 1.5)
    for (row, col), cell in table1.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D6E4F0')

    # Table 2: Alignment comparison
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Alignment Methods Comparison / 对齐方法对比', fontsize=12, fontweight='bold',
                  fontfamily='Microsoft YaHei', pad=10)

    align_data = [
        ['Method/方法', 'Reward Model/奖励模型', 'RL Training/RL训练', 'Complexity/复杂度', 'Best For/最佳适用'],
        ['RLHF/PPO', 'Yes/是', 'Yes/是', 'High/高', 'General alignment/通用对齐'],
        ['DPO', 'No/否', 'No/否', 'Low/低', 'Offline preferences/离线偏好'],
        ['RLAIF', 'AI Judge/AI评判', 'No/否', 'Low/低', 'Scalable alignment/可扩展对齐'],
        ['GRPO', 'No/否', 'Simplified/简化', 'Medium/中', 'Reasoning tasks/推理任务'],
    ]

    table2 = ax2.table(cellText=align_data, loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.5)
    for (row, col), cell in table2.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D6E4F0')

    # Table 3: Evaluation frameworks comparison
    ax3 = axes[2]
    ax3.axis('off')
    ax3.set_title('Evaluation Frameworks Comparison / 评测框架对比', fontsize=12, fontweight='bold',
                  fontfamily='Microsoft YaHei', pad=10)

    eval_data = [
        ['Framework/框架', 'Developer/开发者', 'Tasks/任务', 'Chinese Benchmarks/中文基准', 'Ease of Use/易用性'],
        ['lm-eval-harness', 'EleutherAI', '200+', 'Limited/有限', 'High/高'],
        ['OpenCompass', 'Shanghai AI Lab/上海AI实验室', '100+', 'Strong/强', 'Medium/中'],
        ['HELM', 'Stanford CRFM', '42 scenarios', 'Limited/有限', 'Medium/中'],
        ['HF evaluate', 'Hugging Face', 'Metric API', 'Limited/有限', 'High/高'],
    ]

    table3 = ax3.table(cellText=eval_data, loc='center', cellLoc='center')
    table3.auto_set_font_size(False)
    table3.set_fontsize(9)
    table3.scale(1, 1.5)
    for (row, col), cell in table3.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D6E4F0')

    plt.tight_layout()
    output_path = 'D:/python_code/transformer/Project2_TaskB_Survey/figures/method_comparison_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'已生成: {output_path}')


# ============================================================
# 第二部分：生成 PDF 报告
# ============================================================

class ChinesePDF(FPDF):
    def __init__(self):
        super().__init__()
        # 添加中文字体
        font_path = 'C:/Windows/Fonts/simhei.ttf'
        self.add_font('SimHei', '', font_path, uni=True)
        self.add_font('SimHei', 'B', font_path, uni=True)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font('SimHei', 'B', 10)
        self.cell(0, 10, '大语言模型后训练：微调、对齐与评测综述', 0, 1, 'C')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('SimHei', '', 8)
        self.cell(0, 10, f'第 {self.page_no()} 页', 0, 0, 'C')

    def add_title(self, text, level=1):
        if level == 1:
            self.set_font('SimHei', 'B', 18)
            self.ln(5)
            self.multi_cell(0, 10, text)
            self.ln(3)
        elif level == 2:
            self.set_font('SimHei', 'B', 14)
            self.ln(3)
            self.multi_cell(0, 8, text)
            self.ln(2)
        elif level == 3:
            self.set_font('SimHei', 'B', 12)
            self.ln(2)
            self.multi_cell(0, 7, text)
            self.ln(1)

    def add_text(self, text):
        self.set_font('SimHei', '', 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_code(self, text):
        self.set_font('SimHei', '', 9)
        self.set_fill_color(240, 240, 240)
        for line in text.split('\n'):
            self.cell(0, 6, line, 0, 1, 'L', True)
        self.ln(3)

    def add_table(self, headers, data, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header
        self.set_font('SimHei', 'B', 9)
        self.set_fill_color(68, 114, 196)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, 1, 0, 'C', True)
        self.ln()

        # Data
        self.set_text_color(0, 0, 0)
        self.set_font('SimHei', '', 9)
        for row_idx, row in enumerate(data):
            if row_idx % 2 == 0:
                self.set_fill_color(214, 228, 240)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), 1, 0, 'C', True)
            self.ln()
        self.ln(3)


def generate_pdf():
    """生成完整的 PDF 报告"""
    pdf = ChinesePDF()
    pdf.add_page()

    # ---- 摘要 ----
    pdf.add_title('摘要', level=2)
    pdf.add_text('本报告对大语言模型（LLM）后训练技术进行了全面综述，涵盖三个关键领域：微调（Fine-tuning）、'
                 '对齐（Alignment）和评测框架（Evaluation Harness）。我们考察了从全量微调到参数高效方法'
                 '（如 LoRA 和 QLoRA）的演进，调研了包括 RLHF、DPO 和 GRPO 在内的对齐技术，并回顾了用于'
                 '评估 LLM 能力的主要评测框架。')
    pdf.add_text('关键词：大语言模型、微调、LoRA、QLoRA、RLHF、DPO、GRPO、评测框架')

    # ---- 第一部分：微调 ----
    pdf.add_title('1. 微调（Fine-tuning）', level=2)

    pdf.add_title('1.1 什么是微调？', level=3)
    pdf.add_text('微调是在预训练模型基础上，使用较小的、任务特定的数据集继续训练的过程。其目标是将模型的通用知识'
                 '适配到特定领域或任务，而无需承担从头训练的高昂成本。预训练 LLM 已经学习了丰富的语言表征——句法、'
                 '语义、世界知识和推理模式。微调调整这些表征以在目标任务上表现更好。')

    pdf.add_title('1.2 预训练 vs 微调 vs 指令微调', level=3)
    pdf.add_table(
        ['方面', '预训练', '微调', '指令微调'],
        [
            ['目标', '下一个token预测', '任务特定损失', '响应生成'],
            ['数据', '大规模无标注文本', '任务特定标注数据', '指令-响应对'],
            ['规模', '数十亿token', '数千到数百万', '数千到数百万'],
            ['算力', '数千GPU小时', '数小时到数天', '数小时到数天'],
        ],
        [35, 50, 50, 50]
    )

    pdf.add_title('1.3 全量微调 vs 参数高效微调', level=3)
    pdf.add_text('全量微调更新所有模型参数，对于 7B 参数模型需要约 60-80 GB GPU 显存。参数高效微调（PEFT）'
                 '仅更新一小部分参数，主要包括 LoRA、QLoRA、Prefix Tuning、Adapter Tuning 等方法。')

    pdf.add_title('1.4 LoRA：低秩适配', level=3)
    pdf.add_text('LoRA 基于以下观察：微调过程中的权重更新往往具有低内在秩。将更新分解为两个低秩矩阵：'
                 'W\' = W + ΔW = W + B × A，其中 A ∈ R^(r×d)，B ∈ R^(d×r)，秩 r << d。'
                 '对于 d=4096, r=16：仅需更新 13.1 万参数，而非 1670 万参数。')

    pdf.add_title('1.5 QLoRA：量化 LoRA', level=3)
    pdf.add_text('QLoRA 将 4 位量化与 LoRA 结合：(1) 4 位 NormalFloat（NF4）量化；(2) 双重量化；'
                 '(3) 分页优化器。使得在单块 48GB GPU 上微调 65B 模型成为可能。')

    pdf.add_title('1.6 方法对比', level=3)
    pdf.add_table(
        ['方面', '全量微调', 'LoRA', 'QLoRA'],
        [
            ['可训练参数', '100%', '~0.1-1%', '~0.1-1%'],
            ['GPU显存(7B)', '~60 GB', '~16 GB', '~6 GB'],
            ['训练速度', '基准', '快1.2-1.5倍', '0.8-1.0倍'],
            ['性能', '最佳', '全量微调的98%', '全量微调的95%'],
            ['最低GPU', 'A100 80GB', 'RTX 3090 24GB', 'T4 16GB'],
        ],
        [35, 50, 50, 50]
    )

    # ---- 第二部分：对齐 ----
    pdf.add_title('2. 对齐（Alignment）', level=2)

    pdf.add_title('2.1 什么是对齐？', level=3)
    pdf.add_text('对齐是指确保语言模型的行为与人类价值观和意图保持一致的过程。目标是产生有帮助（Helpful）、'
                 '无害（Harmless）、诚实（Honest）的模型。预训练模型针对下一个 token 预测优化，而非成为有帮助的助手。')

    pdf.add_title('2.2 RLHF：基于人类反馈的强化学习', level=3)
    pdf.add_text('RLHF 是主流的对齐范式，应用于 InstructGPT、ChatGPT 和 Claude。包含三个阶段：\n'
                 '阶段 1：监督微调（SFT）- 在高质量指令-响应对上微调\n'
                 '阶段 2：奖励模型（RM）训练 - 人类标注者排序，训练奖励模型\n'
                 '阶段 3：PPO 优化 - 使用奖励模型优化策略')

    pdf.add_title('2.3 DPO：直接偏好优化', level=3)
    pdf.add_text('DPO 通过直接在偏好数据上优化策略，消除了对独立奖励模型的需求。关键洞察是 RLHF 下的最优策略'
                 '具有闭式解。相比 RLHF：无需奖励模型、无需 RL 训练循环、更简单稳定、计算成本更低。')

    pdf.add_title('2.4 GRPO：群组相对策略优化', level=3)
    pdf.add_text('GRPO 是 DeepSeek-R1 中使用的较新对齐方法，通过移除评论家/价值模型简化 PPO。'
                 '对每个提示采样 G 个响应，通过组内归一化计算优势。优势：无需价值模型（节省约 50% 显存），'
                 '对推理任务有效。')

    pdf.add_title('2.5 对齐方法对比', level=3)
    pdf.add_table(
        ['方法', '奖励模型', 'RL训练', '复杂度', '最佳适用'],
        [
            ['RLHF/PPO', '是', '是', '高', '通用对齐'],
            ['DPO', '否', '否', '低', '离线偏好'],
            ['RLAIF', 'AI评判', '否', '低', '可扩展对齐'],
            ['GRPO', '否', '简化', '中', '推理任务'],
        ],
        [30, 30, 30, 30, 60]
    )

    # ---- 第三部分：评测框架 ----
    pdf.add_title('3. 评测框架（Evaluation Harness）', level=2)

    pdf.add_title('3.1 什么是评测框架？', level=3)
    pdf.add_text('评测框架是一个标准化系统，用于在多个基准测试、任务和指标上系统地评估语言模型。'
                 '它提供一致性、可复现性、可比性和自动化能力。')

    pdf.add_title('3.2 常见基准测试', level=3)
    pdf.add_table(
        ['基准测试', '能力', '语言', '指标'],
        [
            ['MMLU', '知识(57学科)', '英文', '准确率'],
            ['HellaSwag', '常识推理', '英文', '准确率'],
            ['ARC', '科学问答', '英文', '准确率'],
            ['GSM8K', '数学推理', '英文', '准确率'],
            ['HumanEval', '代码生成', 'Python', 'Pass@k'],
            ['C-Eval', '中文知识', '中文', '准确率'],
        ],
        [40, 55, 40, 40]
    )

    pdf.add_title('3.3 评测框架对比', level=3)
    pdf.add_table(
        ['框架', '开发者', '任务数', '中文基准', '易用性'],
        [
            ['lm-eval-harness', 'EleutherAI', '200+', '有限', '高'],
            ['OpenCompass', '上海AI实验室', '100+', '强', '中'],
            ['HELM', 'Stanford CRFM', '42场景', '有限', '中'],
        ],
        [35, 40, 30, 35, 35]
    )

    pdf.add_title('3.4 lm-evaluation-harness 使用', level=3)
    pdf.add_code('lm_eval --model hf \\\n'
                 '    --model_args pretrained=MODEL_NAME \\\n'
                 '    --tasks mmlu,hellaswag \\\n'
                 '    --device cuda:0 \\\n'
                 '    --batch_size 8')

    # ---- 第四部分：后训练流程 ----
    pdf.add_title('4. 后训练流程', level=2)
    pdf.add_text('完整的 LLM 后训练流程：预训练模型 → 监督微调（SFT）→ 偏好数据收集 → '
                 '对齐训练（RLHF/DPO/GRPO）→ 评测（lm-eval-harness/OpenCompass/HELM）→ '
                 '基准测试结果与错误分析。每个阶段建立在前一阶段的基础上，逐步将原始语言模型转变为'
                 '有能力、安全且可靠的助手。')

    # ---- 第五部分：结论 ----
    pdf.add_title('5. 结论', level=2)
    pdf.add_text('本综述涵盖了 LLM 后训练的三大支柱：\n'
                 '1. 微调将预训练模型适配到特定任务，LoRA 和 QLoRA 以最小成本实现了高效适配\n'
                 '2. 对齐确保模型按照人类价值观行事，DPO 为传统 RLHF 提供了更简单的替代方案\n'
                 '3. 评测框架为衡量模型能力提供了标准化系统\n\n'
                 '该领域正在快速发展，GRPO 等新方法正在推动对齐技术的边界，特别是在推理任务方面。')

    # ---- 参考文献 ----
    pdf.add_title('参考文献', level=2)
    refs = [
        '[1] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021',
        '[2] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023',
        '[3] Ouyang et al., "Training language models to follow instructions with human feedback", 2022',
        '[4] Rafailov et al., "Direct Preference Optimization", 2023',
        '[5] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning", 2024',
        '[6] Taori et al., "Stanford Alpaca", 2023',
        '[7] Liang et al., "Holistic Evaluation of Language Models (HELM)", 2022',
        '[8] Hendrycks et al., "Measuring Massive Multitask Language Understanding (MMLU)", 2021',
        '[9] Gao et al., "lm-evaluation-harness", 2023',
        '[10] Vaswani et al., "Attention Is All You Need", 2017',
    ]
    for ref in refs:
        pdf.add_text(ref)

    # 保存 PDF
    output_path = 'D:/python_code/transformer/Project2_TaskB_Survey/report/project_report.pdf'
    pdf.output(output_path)
    print(f'已生成: {output_path}')


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print('开始生成资源...')
    print()

    print('1. 生成后训练流程图...')
    create_post_training_pipeline()

    print('2. 生成方法对比表...')
    create_method_comparison_table()

    print('3. 生成 PDF 报告...')
    generate_pdf()

    print()
    print('全部完成！')
