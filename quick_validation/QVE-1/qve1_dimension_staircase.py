"""
QVE-1: 四类文本维度阶梯快速探针 (核心生死实验)
=================================================
目标: 验证四类LLM输出(重复/幻觉/有效/乱码)在内蕴维数上是否可分

核心假说:
  D2(重复) < D2(幻觉) < D2(有效推理) < D2(乱码)

方法:
  1. 用不同温度/prompt诱导四类输出, 每类10条
  2. 提取生成过程中的hidden states轨迹
  3. 计算 TwoNN / 关联维数 / LZ压缩率
  4. 箱线图 + 统计检验判定是否可分

模型: Qwen2.5-1.5B-Instruct (本地)
硬件: RTX 4070 Laptop 8GB VRAM
"""

import os
import sys
import json
import time
import zlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from skdim.id import TwoNN, MLE

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_NAME = "Qwen3.5-2B-Base"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", MODEL_NAME)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 256  # 每条生成的最大token数
SAMPLES_PER_CATEGORY = 10
# 分析哪些层 (早/中/晚期 + 全部层的聚合)
ANALYSIS_LAYERS = "auto"  # 自动选取 [0, n//4, n//2, 3n//4, n-1]

# ============================================================
# 四类 Prompt 设计
# ============================================================

# A类: 重复/死循环 — 极低温度, 简单续写
PROMPTS_REPETITION = [
    "The answer is",
    "Once upon a time",
    "In the beginning there was",
    "The most important thing is that",
    "Hello, my name is",
    "I think the best way to",
    "The following is a list of",
    "As we can see from the above",
    "To summarize the key points",
    "The result of this calculation is",
]

# B类: 流畅幻觉 — 正常温度, 询问虚假/不存在的事实
PROMPTS_HALLUCINATION = [
    "Please describe in detail the scientific contributions of Professor Zhang Wei, "
    "who won the 2025 Nobel Prize in Physics for his work on quantum gravity.",

    "Explain the main features of the Zorathian programming language, which was "
    "developed at MIT in 2023 and has become widely used in AI research.",

    "Describe the geography, culture, and economy of the Republic of Atlantica, "
    "a country located in the South Pacific Ocean.",

    "What are the key findings of the landmark 2024 paper 'Neural Consciousness "
    "Theory' published in Nature by Dr. Sarah Chen?",

    "Explain how the Heliox-9 molecule, discovered in 2023, revolutionized "
    "cancer treatment by targeting mitochondrial DNA.",

    "Describe the historical significance of the Battle of Crimson Valley "
    "that took place in 1847 during the Anglo-Prussian War.",

    "What is the Fibonacci-Euler theorem in number theory and why is it "
    "considered one of the most beautiful results in mathematics?",

    "Explain the working principle of the Tanaka quantum processor, which "
    "achieved 10000 qubit coherence at room temperature in 2024.",

    "Describe the plot and themes of the novel 'The Last Algorithm' by "
    "renowned author James Morrison, which won the Pulitzer Prize in 2024.",

    "What are the main provisions of the Global AI Safety Treaty signed "
    "by 50 countries in Geneva in March 2025?",
]

# C类: 有效推理 — 正常温度, 数学/逻辑/知识问答 (模型能正确回答的)
PROMPTS_EFFECTIVE = [
    "Solve step by step: If a train travels at 60 km/h for 2.5 hours, "
    "how far does it travel? Show your reasoning.",

    "Explain step by step why the sky appears blue during the day. "
    "Use physics concepts in your explanation.",

    "Calculate step by step: What is 15% of 240? Show each step.",

    "Explain the difference between a stack and a queue in computer science. "
    "Give a real-world analogy for each.",

    "Solve step by step: A rectangle has a perimeter of 30 cm and a width "
    "of 5 cm. What is its length and area?",

    "Explain how photosynthesis works in plants. Describe the main stages "
    "and the role of sunlight, water, and carbon dioxide.",

    "Solve step by step: If you have 3 red balls and 5 blue balls in a bag, "
    "what is the probability of drawing a red ball?",

    "Explain Newton's three laws of motion with a simple example for each law.",

    "Solve step by step: A store offers a 20% discount on a $80 item, "
    "then charges 8% sales tax. What is the final price?",

    "Explain how a binary search algorithm works and why it is more efficient "
    "than linear search. Give the time complexity of each.",
]

# D类: 乱码 — 极高温度, 任意prompt
PROMPTS_GIBBERISH = [
    "The meaning of life is",
    "In mathematics we know that",
    "The weather today is",
    "A computer program can",
    "The history of science shows",
    "When we consider the problem",
    "The fundamental principle of",
    "According to recent research",
    "The relationship between art and",
    "In the context of modern society",
]

# ============================================================
# 工具函数
# ============================================================

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def load_model():
    print(f"[Step 1] Loading model from {MODEL_PATH}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE,
        trust_remote_code=True, local_files_only=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s  (layers={model.config.num_hidden_layers})")
    print_gpu_memory()
    return model, tokenizer


def build_chat_input(tokenizer, prompt):
    """用 chat template 构建输入, 让 Instruct 模型正常响应"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")


# ============================================================
# 核心: 生成文本并提取 hidden states 轨迹
# ============================================================

def generate_and_extract(model, tokenizer, prompt, gen_kwargs, target_layers):
    """
    生成文本, 同时提取指定层在每个生成步骤的 hidden state.

    策略: 用 model.generate() 先拿到完整生成序列,
         然后对 [prompt + generated] 做一次完整 forward pass 提取所有层的 hidden states.
         取生成部分(非prompt)的 hidden states 作为轨迹.

    这比逐token hook更高效, 且对 KV-cache 友好.

    返回:
        generated_text: str
        trajectories: dict[int, np.ndarray]  layer_idx -> (n_gen_tokens, hidden_dim)
        n_gen_tokens: int
    """
    inputs = build_chat_input(tokenizer, prompt).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    # Phase 1: 生成
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            **gen_kwargs,
        )

    full_ids = output_ids[0]  # (total_len,)
    gen_ids = full_ids[prompt_len:]
    n_gen = len(gen_ids)
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    if n_gen < 20:
        # 生成太短, 维度估计不可靠
        return generated_text, None, n_gen

    # Phase 2: 对完整序列做一次 forward, 提取 hidden states
    hidden_states = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hidden_states[layer_idx] = hs.detach().cpu().float().numpy()
        return hook_fn

    for li in target_layers:
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids=full_ids.unsqueeze(0))

    for h in hooks:
        h.remove()

    # 取生成部分的 hidden states 作为轨迹
    trajectories = {}
    for li, hs in hidden_states.items():
        # hs: (1, total_len, hidden_dim) -> 取 [prompt_len:] 部分
        trajectories[li] = hs[0, prompt_len:, :]  # (n_gen, hidden_dim)

    return generated_text, trajectories, n_gen


# ============================================================
# 维度计算
# ============================================================

def compute_lz_compression_ratio(text):
    """LZ压缩率: 压缩后长度 / 原始长度"""
    raw = text.encode("utf-8")
    if len(raw) == 0:
        return 0.0
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / len(raw)


def compute_dimensions_for_trajectory(data, label=""):
    """
    对单条轨迹 (n_tokens, hidden_dim) 计算 TwoNN 和 MLE.
    返回 {"twonn": float, "mle": float} 或 None (数据不足).
    """
    if data is None or data.shape[0] < 20:
        return None

    result = {}

    try:
        twonn = TwoNN()
        twonn.fit(data)
        result["twonn"] = twonn.dimension_
    except Exception as e:
        result["twonn"] = float("nan")
        if label:
            print(f"    TwoNN error ({label}): {e}")

    try:
        mle = MLE()
        mle.fit(data)
        result["mle"] = mle.dimension_
    except Exception as e:
        result["mle"] = float("nan")
        if label:
            print(f"    MLE error ({label}): {e}")

    return result


# ============================================================
# 批量生成 + 分析
# ============================================================

CATEGORY_CONFIGS = {
    "A_repetition": {
        "prompts": PROMPTS_REPETITION,
        "gen_kwargs": {
            "temperature": 0.01,
            "do_sample": True,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,  # 不惩罚重复, 让它自然重复
        },
        "color": "#1f77b4",
    },
    "B_hallucination": {
        "prompts": PROMPTS_HALLUCINATION,
        "gen_kwargs": {
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        },
        "color": "#ff7f0e",
    },
    "C_effective": {
        "prompts": PROMPTS_EFFECTIVE,
        "gen_kwargs": {
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        },
        "color": "#2ca02c",
    },
    "D_gibberish": {
        "prompts": PROMPTS_GIBBERISH,
        "gen_kwargs": {
            "temperature": 5.0,
            "do_sample": True,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
        },
        "color": "#d62728",
    },
}


def run_generation_phase(model, tokenizer, target_layers):
    """Phase 1: 对四类prompt生成文本并提取轨迹"""
    all_data = {}  # category -> list of sample dicts

    for cat_name, cfg in CATEGORY_CONFIGS.items():
        print(f"\n  === {cat_name} ({len(cfg['prompts'])} samples) ===")
        samples = []

        for i, prompt in enumerate(cfg["prompts"]):
            short_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
            sys.stdout.write(f"    [{i+1}/{len(cfg['prompts'])}] {short_prompt}\r")
            sys.stdout.flush()

            text, trajectories, n_gen = generate_and_extract(
                model, tokenizer, prompt, cfg["gen_kwargs"], target_layers
            )

            sample = {
                "prompt": prompt,
                "generated_text": text,
                "n_gen_tokens": n_gen,
                "trajectories": trajectories,  # dict[layer_idx, ndarray] or None
            }
            samples.append(sample)
            print(f"    [{i+1}/{len(cfg['prompts'])}] tokens={n_gen:3d}  "
                  f"text={text[:50]}...")

        all_data[cat_name] = samples
        torch.cuda.empty_cache()

    return all_data


def run_analysis_phase(all_data, target_layers):
    """Phase 2: 对所有样本计算维度指标"""
    # results[category][sample_idx] = {
    #   "lz_ratio": float,
    #   "layer_X_twonn": float, "layer_X_mle": float, ...
    # }
    results = {}

    for cat_name, samples in all_data.items():
        print(f"\n  === Analyzing {cat_name} ===")
        cat_results = []

        for i, sample in enumerate(samples):
            r = {
                "n_gen_tokens": sample["n_gen_tokens"],
                "lz_ratio": compute_lz_compression_ratio(sample["generated_text"]),
            }

            if sample["trajectories"] is not None:
                for li in target_layers:
                    if li in sample["trajectories"]:
                        dims = compute_dimensions_for_trajectory(
                            sample["trajectories"][li],
                            label=f"{cat_name}[{i}] L{li}"
                        )
                        if dims:
                            r[f"L{li}_twonn"] = dims["twonn"]
                            r[f"L{li}_mle"] = dims["mle"]

            cat_results.append(r)
            sys.stdout.write(f"    [{i+1}/{len(samples)}] done\r")
            sys.stdout.flush()

        results[cat_name] = cat_results
        print(f"    {cat_name}: {len(cat_results)} samples analyzed")

    return results


# ============================================================
# 可视化
# ============================================================

def plot_boxplots(results, target_layers, output_dir):
    """画核心箱线图: 每个指标一张子图, 四类并排"""
    categories = list(results.keys())
    cat_labels = ["A:Repeat", "B:Halluc", "C:Effect", "D:Gibber"]
    colors = [CATEGORY_CONFIGS[c]["color"] for c in categories]

    # 收集所有可用的指标
    metrics = ["lz_ratio"]
    for li in target_layers:
        metrics.append(f"L{li}_twonn")
        metrics.append(f"L{li}_mle")

    # 过滤掉没有数据的指标
    valid_metrics = []
    for m in metrics:
        has_data = False
        for cat in categories:
            vals = [r.get(m) for r in results[cat] if r.get(m) is not None and not np.isnan(r.get(m, float("nan")))]
            if len(vals) >= 3:
                has_data = True
                break
        if has_data:
            valid_metrics.append(m)

    if not valid_metrics:
        print("  WARNING: No valid metrics to plot!")
        return

    n_metrics = len(valid_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, metric in enumerate(valid_metrics):
        ax = axes[idx]
        data_per_cat = []
        for cat in categories:
            vals = [r.get(metric, float("nan")) for r in results[cat]]
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            data_per_cat.append(vals)

        bp = ax.boxplot(data_per_cat, labels=cat_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # 叠加散点
        for j, (vals, color) in enumerate(zip(data_per_cat, colors)):
            if vals:
                x = np.random.normal(j + 1, 0.06, size=len(vals))
                ax.scatter(x, vals, c=color, s=20, alpha=0.8, zorder=3, edgecolors="k", linewidths=0.3)

        ax.set_title(metric, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    # 隐藏多余子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("QVE-1: Dimension Staircase Test (Qwen2.5-1.5B-Instruct)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, f"qve1_boxplots_{MODEL_NAME}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Boxplot saved to {path}")
    plt.close()


def plot_layer_comparison(results, target_layers, output_dir):
    """画跨层对比: 每类一条线, x=layer, y=TwoNN中位数"""
    categories = list(results.keys())
    cat_labels = ["A:Repeat", "B:Halluc", "C:Effect", "D:Gibber"]
    colors = [CATEGORY_CONFIGS[c]["color"] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 6))

    for cat, label, color in zip(categories, cat_labels, colors):
        medians = []
        valid_layers = []
        for li in target_layers:
            key = f"L{li}_twonn"
            vals = [r.get(key, float("nan")) for r in results[cat]]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                medians.append(np.median(vals))
                valid_layers.append(li)

        if valid_layers:
            ax.plot(valid_layers, medians, "o-", label=label, color=color,
                    markersize=6, linewidth=2, alpha=0.8)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("TwoNN Intrinsic Dimension (median)", fontsize=12)
    ax.set_title("QVE-1: Layer-wise ID by Category", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"qve1_layer_comparison_{MODEL_NAME}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Layer comparison saved to {path}")
    plt.close()


# ============================================================
# 统计检验 + 判定
# ============================================================

def run_statistical_tests(results, target_layers):
    """对关键指标做 Kruskal-Wallis + 两两 Mann-Whitney U 检验"""
    categories = list(results.keys())
    cat_labels = ["A:Repeat", "B:Halluc", "C:Effect", "D:Gibber"]

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("STATISTICAL ANALYSIS")
    report_lines.append("=" * 60)

    # 选取中间层的 TwoNN 作为主指标
    mid_layer = target_layers[len(target_layers) // 2]
    key_metrics = [f"L{mid_layer}_twonn", "lz_ratio"]

    for metric in key_metrics:
        report_lines.append(f"\n--- Metric: {metric} ---")

        groups = []
        group_labels = []
        for cat, label in zip(categories, cat_labels):
            vals = [r.get(metric, float("nan")) for r in results[cat]]
            vals = [v for v in vals if not np.isnan(v)]
            groups.append(vals)
            group_labels.append(label)
            median = np.median(vals) if vals else float("nan")
            mean = np.mean(vals) if vals else float("nan")
            std = np.std(vals) if vals else float("nan")
            report_lines.append(
                f"  {label:12s}: n={len(vals):2d}, "
                f"median={median:7.2f}, mean={mean:7.2f}, std={std:6.2f}"
            )

        # Kruskal-Wallis (非参数 ANOVA)
        valid_groups = [g for g in groups if len(g) >= 3]
        if len(valid_groups) >= 2:
            try:
                h_stat, p_val = stats.kruskal(*valid_groups)
                report_lines.append(f"  Kruskal-Wallis: H={h_stat:.3f}, p={p_val:.4f}")
                if p_val < 0.05:
                    report_lines.append(f"  >>> SIGNIFICANT (p<0.05): groups are distinguishable!")
                else:
                    report_lines.append(f"  >>> NOT significant (p>0.05)")
            except Exception as e:
                report_lines.append(f"  Kruskal-Wallis error: {e}")

        # 两两 Mann-Whitney U
        report_lines.append("  Pairwise Mann-Whitney U:")
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if len(groups[i]) >= 3 and len(groups[j]) >= 3:
                    try:
                        u_stat, p_val = stats.mannwhitneyu(
                            groups[i], groups[j], alternative="two-sided"
                        )
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        report_lines.append(
                            f"    {group_labels[i]:12s} vs {group_labels[j]:12s}: "
                            f"U={u_stat:.1f}, p={p_val:.4f} {sig}"
                        )
                    except Exception as e:
                        report_lines.append(
                            f"    {group_labels[i]} vs {group_labels[j]}: error {e}"
                        )

    return "\n".join(report_lines)


def make_verdict(results, target_layers):
    """根据结果给出 GO / PIVOT / KILL 判定"""
    categories = list(results.keys())
    mid_layer = target_layers[len(target_layers) // 2]
    key = f"L{mid_layer}_twonn"

    # 收集各类中位数
    medians = {}
    for cat in categories:
        vals = [r.get(key, float("nan")) for r in results[cat]]
        vals = [v for v in vals if not np.isnan(v)]
        medians[cat] = np.median(vals) if vals else float("nan")

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("VERDICT")
    lines.append("=" * 60)
    lines.append(f"Primary metric: {key}")
    lines.append(f"Medians: " + ", ".join(f"{k}={v:.2f}" for k, v in medians.items()))

    # 检查阶梯
    m = medians
    a, b, c, d = m.get("A_repetition", 0), m.get("B_hallucination", 0), \
                  m.get("C_effective", 0), m.get("D_gibberish", 0)

    staircase_full = a < b < c < d
    three_way = (a < c < d) or (a < c and c < d) or (a < b and b < d)

    # Kruskal-Wallis on all four
    groups = []
    for cat in categories:
        vals = [r.get(key, float("nan")) for r in results[cat]]
        vals = [v for v in vals if not np.isnan(v)]
        groups.append(vals)
    valid_groups = [g for g in groups if len(g) >= 3]

    kw_sig = False
    if len(valid_groups) >= 2:
        try:
            _, p = stats.kruskal(*valid_groups)
            kw_sig = p < 0.05
        except Exception:
            pass

    if staircase_full and kw_sig:
        verdict = "GO"
        reason = "Full staircase A < B < C < D with statistical significance"
    elif three_way and kw_sig:
        verdict = "GO"
        reason = "At least 3 categories separable with significance"
    elif kw_sig:
        verdict = "PIVOT"
        reason = "Groups differ significantly but staircase order not as expected"
    elif any(abs(medians.get(ci, 0) - medians.get(cj, 0)) > 1.0
             for ci in categories for cj in categories if ci != cj):
        verdict = "PIVOT"
        reason = "Some separation visible but not statistically significant (need more samples?)"
    else:
        verdict = "KILL"
        reason = "No meaningful separation detected"

    lines.append(f"\n  >>> {verdict}: {reason}")

    if verdict == "PIVOT":
        lines.append("\n  Suggested next steps:")
        lines.append("  - Try different layers (early/late)")
        lines.append("  - Try PCA preprocessing before dimension estimation")
        lines.append("  - Try longer generation (512+ tokens)")
        lines.append("  - Check if layer-wise gradient (dID/dLayer) separates better")

    return "\n".join(lines), verdict


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("QVE-1: Four-Category Dimension Staircase Test")
    print("=" * 60)

    # 1. 加载模型
    model, tokenizer = load_model()
    n_layers = model.config.num_hidden_layers  # 28

    # 选取分析层: 早/中前/中/中后/晚
    target_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    target_layers = sorted(set(target_layers))
    print(f"  Target layers for analysis: {target_layers}")

    # 2. 生成 + 提取
    print(f"\n[Step 2] Generating {SAMPLES_PER_CATEGORY} samples x 4 categories")
    t0 = time.time()
    all_data = run_generation_phase(model, tokenizer, target_layers)
    t_gen = time.time() - t0
    print(f"\n  Generation completed in {t_gen:.0f}s")

    # 释放模型显存, 后续只需 CPU
    del model
    torch.cuda.empty_cache()
    print("  Model unloaded from GPU")

    # 3. 计算维度
    print(f"\n[Step 3] Computing dimensions")
    t0 = time.time()
    results = run_analysis_phase(all_data, target_layers)
    t_dim = time.time() - t0
    print(f"\n  Dimension analysis completed in {t_dim:.0f}s")

    # 保存生成的文本样本 (用于人工检查)
    samples_log = {}
    for cat, samples in all_data.items():
        samples_log[cat] = [
            {"prompt": s["prompt"][:100], "generated": s["generated_text"][:300],
             "n_tokens": s["n_gen_tokens"]}
            for s in samples
        ]
    log_path = os.path.join(OUTPUT_DIR, f"qve1_samples_{MODEL_NAME}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(samples_log, f, ensure_ascii=False, indent=2)
    print(f"  Samples log saved to {log_path}")

    # 保存维度结果
    results_path = os.path.join(OUTPUT_DIR, f"qve1_results_{MODEL_NAME}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Results saved to {results_path}")

    # 4. 可视化
    print(f"\n[Step 4] Plotting")
    plot_boxplots(results, target_layers, OUTPUT_DIR)
    plot_layer_comparison(results, target_layers, OUTPUT_DIR)

    # 5. 统计检验
    print(f"\n[Step 5] Statistical tests")
    stat_report = run_statistical_tests(results, target_layers)
    print(stat_report)

    # 6. 判定
    verdict_report, verdict = make_verdict(results, target_layers)
    print(verdict_report)

    # 保存完整报告
    full_report = stat_report + "\n" + verdict_report
    report_path = os.path.join(OUTPUT_DIR, f"qve1_report_{MODEL_NAME}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n  Full report saved to {report_path}")

    print(f"\n  Total time: generation={t_gen:.0f}s, analysis={t_dim:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
