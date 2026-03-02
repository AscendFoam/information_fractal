"""
QVE-0: 工程流程端到端验证
=============================
目标: 验证 hidden states 提取 + 内蕴维数计算 的完整 pipeline 可行性

模型: Qwen2.5-1.5B-Instruct (本地, 28层, hidden_size=1536)
硬件: RTX 4070 Laptop 8GB VRAM

验证项:
  1. 模型加载到 GPU (bf16, ~3GB显存)
  2. Forward hook 提取每层 hidden states
  3. scikit-dimension TwoNN 计算内蕴维数
  4. 画出 Layer vs ID 曲线
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from skdim.id import TwoNN, MLE

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.path.dirname(__file__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 测试用文本: 一段有意义的英文 + 一段中文, 确保 token 数足够
TEST_TEXTS = [
    # 自然语言段落 (英文)
    (
        "The theory of fractal geometry, developed by Benoit Mandelbrot in the 1970s, "
        "provides a mathematical framework for describing complex shapes and patterns "
        "found in nature. Unlike traditional Euclidean geometry, which deals with smooth "
        "shapes like circles and rectangles, fractal geometry captures the roughness and "
        "self-similarity observed in coastlines, mountains, clouds, and even the branching "
        "patterns of trees and blood vessels. A key concept in fractal geometry is the "
        "fractal dimension, which quantifies how a pattern fills space across different "
        "scales. This dimension is often a non-integer value, reflecting the intermediate "
        "complexity between simple lines and filled planes."
    ),
    # 自然语言段落 (中文)
    (
        "大型语言模型的涌现能力是当前人工智能研究的核心话题之一。"
        "当模型参数量超过某个临界阈值时，模型会突然展现出在小规模时完全不具备的能力，"
        "例如思维链推理、少样本学习和代码生成。这种现象类似于物理学中的相变，"
        "即系统在某个临界点附近发生质的飞跃。理解涌现能力的本质，"
        "对于预测更大规模模型的行为以及确保人工智能安全具有重要意义。"
    ),
]


def print_gpu_memory():
    """打印当前 GPU 显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def load_model_and_tokenizer():
    """加载模型和分词器"""
    print(f"[1/4] Loading model from {MODEL_PATH}")
    print(f"  Device: {DEVICE}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    t1 = time.time()
    print(f"  Loaded in {t1 - t0:.1f}s")
    print(f"  Model layers: {model.config.num_hidden_layers}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print_gpu_memory()
    return model, tokenizer


def extract_hidden_states(model, tokenizer, text):
    """
    通过 forward hook 提取每一层的 hidden states

    返回: dict[int, np.ndarray]  layer_idx -> (n_tokens, hidden_dim)
    """
    hidden_states = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Qwen2DecoderLayer 的 output 是 tuple, 第一个元素是 hidden state
            hs = output[0] if isinstance(output, tuple) else output
            hidden_states[layer_idx] = hs.detach().cpu().float().numpy()
        return hook_fn

    # 注册 hook 到每一层 decoder layer
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Forward pass
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    n_tokens = inputs["input_ids"].shape[1]
    print(f"  Input tokens: {n_tokens}")

    with torch.no_grad():
        _ = model(**inputs)

    # 清理 hooks
    for h in hooks:
        h.remove()

    # 转换: squeeze batch dim
    result = {}
    for layer_idx, hs in hidden_states.items():
        result[layer_idx] = hs.squeeze(0)  # (n_tokens, hidden_dim)

    return result, n_tokens


def compute_intrinsic_dimensions(hidden_states_dict, min_points=10):
    """
    对每层的 hidden states 计算内蕴维数 (TwoNN + MLE)

    参数:
        hidden_states_dict: dict[int, np.ndarray], layer -> (n_tokens, hidden_dim)
        min_points: TwoNN 需要的最少数据点数

    返回:
        dict with layer -> {"twonn": float, "mle": float}
    """
    results = {}
    for layer_idx in sorted(hidden_states_dict.keys()):
        data = hidden_states_dict[layer_idx]
        n_points, dim = data.shape

        if n_points < min_points:
            print(f"  Layer {layer_idx}: skipped (only {n_points} points, need {min_points})")
            continue

        try:
            twonn = TwoNN()
            twonn.fit(data)
            id_twonn = twonn.dimension_
        except Exception as e:
            id_twonn = float("nan")
            print(f"  Layer {layer_idx} TwoNN error: {e}")

        try:
            mle = MLE()
            mle.fit(data)
            id_mle = mle.dimension_
        except Exception as e:
            id_mle = float("nan")
            print(f"  Layer {layer_idx} MLE error: {e}")

        results[layer_idx] = {"twonn": id_twonn, "mle": id_mle}

    return results


def plot_layer_id_curve(all_results, output_path):
    """
    画 Layer vs ID 曲线

    参数:
        all_results: dict[str, dict[int, {"twonn": float, "mle": float}]]
            text_label -> layer -> dimensions
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for label, results in all_results.items():
        layers = sorted(results.keys())
        twonn_vals = [results[l]["twonn"] for l in layers]
        mle_vals = [results[l]["mle"] for l in layers]

        axes[0].plot(layers, twonn_vals, "o-", label=label, markersize=3, alpha=0.8)
        axes[1].plot(layers, mle_vals, "o-", label=label, markersize=3, alpha=0.8)

    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Intrinsic Dimension")
    axes[0].set_title("TwoNN Estimator")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Intrinsic Dimension")
    axes[1].set_title("MLE Estimator")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("QVE-0: Layer-wise Intrinsic Dimension (Qwen2.5-1.5B-Instruct)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to {output_path}")


def main():
    print("=" * 60)
    print("QVE-0: Engineering Pipeline Validation")
    print("=" * 60)

    # Step 1: 加载模型
    model, tokenizer = load_model_and_tokenizer()

    # Step 2: 提取 hidden states
    print(f"\n[2/4] Extracting hidden states from {len(TEST_TEXTS)} texts")
    all_hidden = {}
    for i, text in enumerate(TEST_TEXTS):
        label = f"Text_{i+1} ({'EN' if i == 0 else 'ZH'})"
        print(f"\n  --- {label} ---")
        hs_dict, n_tokens = extract_hidden_states(model, tokenizer, text)
        print(f"  Extracted {len(hs_dict)} layers, shape per layer: ({n_tokens}, {model.config.hidden_size})")
        all_hidden[label] = hs_dict

    print_gpu_memory()

    # Step 3: 计算内蕴维数
    print(f"\n[3/4] Computing intrinsic dimensions")
    all_results = {}
    for label, hs_dict in all_hidden.items():
        print(f"\n  --- {label} ---")
        t0 = time.time()
        results = compute_intrinsic_dimensions(hs_dict)
        t1 = time.time()
        print(f"  Computed in {t1 - t0:.1f}s")
        all_results[label] = results

        # 打印摘要
        layers = sorted(results.keys())
        twonn_vals = [results[l]["twonn"] for l in layers]
        mle_vals = [results[l]["mle"] for l in layers]
        print(f"  TwoNN range: [{min(twonn_vals):.2f}, {max(twonn_vals):.2f}], mean={np.mean(twonn_vals):.2f}")
        print(f"  MLE   range: [{min(mle_vals):.2f}, {max(mle_vals):.2f}], mean={np.mean(mle_vals):.2f}")

    # Step 4: 可视化
    print(f"\n[4/4] Plotting Layer vs ID curve")
    plot_path = os.path.join(OUTPUT_DIR, "qve0_layer_id_curve.png")
    plot_layer_id_curve(all_results, plot_path)

    # 总结
    print("\n" + "=" * 60)
    print("QVE-0 SUMMARY")
    print("=" * 60)
    print(f"  Model loaded:          OK")
    print(f"  Hidden states extracted: OK ({len(all_hidden)} texts)")
    print(f"  ID computed:           OK ({len(all_results)} texts x {len(next(iter(all_results.values())))} layers)")
    print(f"  Plot saved:            {plot_path}")
    print(f"\n  VERDICT: Pipeline is functional. Ready for QVE-1.")


if __name__ == "__main__":
    main()
