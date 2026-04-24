#!/usr/bin/env python3
"""
check_regression.py — CNN 精度回归测试
======================================

防止"优化后反而更差"的问题。用法：

    # 1. 记录当前基准（首次运行，或在已知好的模型上运行）
    python check_regression.py --save-baseline

    # 2. 检查当前模型是否退化（每次改代码/训练后运行）
    python check_regression.py

    # 3. 自定义容差
    python check_regression.py --tolerance 0.15  # 允许 MAE 恶化 15%

原理：
    加载 best_model.pt，在测试集上评估，与保存的 baseline 指标对比。
    如果 MAE 恶化超过 tolerance（默认 10%），返回非零退出码。

未来让其他 AI 改代码时，在你的 CLAUDE.md 或提示词中加上：
    "每次修改 train_cnn.py 或相关模块后，必须运行 python check_regression.py
     确保测试通过后再提交。"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from angle_cnn.core.config import THETA_MAX, THETA_MIN
from angle_cnn.core.eval_utils import load_model_from_checkpoint
from angle_cnn.core.io_utils import load_npz_dataset
from angle_cnn.core.metrics import compute_stratified_metrics

DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
DATASET_PATH = os.path.join(DATA_DIR, "moire_dataset.npz")
MODEL_PATH = os.path.join(OUT_DIR, "best_model.pt")
BASELINE_PATH = os.path.join(OUT_DIR, "regression_baseline.json")


def evaluate_model(model, images, labels, device, batch_size=128):
    """Evaluate model on test set, return (predictions_deg, labels_deg)."""
    import torch

    model.eval()
    all_preds, all_labels = [], []
    n = len(labels)

    for i in range(0, n, batch_size):
        batch_imgs = images[i : i + batch_size]
        batch_lbls = labels[i : i + batch_size]

        if batch_imgs.ndim == 3:
            x = torch.from_numpy(batch_imgs).unsqueeze(1).float().to(device)
        else:
            x = torch.from_numpy(batch_imgs).float().to(device)

        with torch.no_grad():
            preds_norm = model(x).squeeze(1).clamp(0, 1).cpu().numpy()

        preds_deg = preds_norm * (THETA_MAX - THETA_MIN) + THETA_MIN
        lbls_deg = batch_lbls

        all_preds.append(preds_deg)
        all_labels.append(lbls_deg)

    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(description="CNN 精度回归测试")
    parser.add_argument("--save-baseline", action="store_true",
                        help="保存当前模型指标作为基准")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="允许 MAE 恶化的比例（默认 10%%）")
    parser.add_argument("--force-update", action="store_true",
                        help="强制更新基准（即使当前更差）")
    cli = parser.parse_args()

    print("=" * 50)
    print("CNN 精度回归测试")
    print("=" * 50)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # 加载模型和数据
    model, meta = load_model_from_checkpoint(MODEL_PATH, device)
    data = load_npz_dataset(DATASET_PATH)
    images_test = data["images_test"]
    labels_test = data["labels_test"]

    print(f"模型: {MODEL_PATH}")
    print(f"  arch={meta.get('arch')}, channels={meta.get('n_channels')}, "
          f"version={meta.get('model_version', '?')}")
    print(f"测试集: {len(labels_test)} 样本")
    print(f"设备: {device}\n")

    # 评估
    preds, labels = evaluate_model(model, images_test, labels_test, device)
    metrics = compute_stratified_metrics(preds, labels)

    current = {
        "test_mae": float(metrics.overall_mae),
        "p90": float(metrics.percentile_errors.get(90, 0)),
        "p95": float(metrics.percentile_errors.get(95, 0)),
        "p99": float(metrics.percentile_errors.get(99, 0)),
        "small_angle_mae": float(metrics.small_angle_mae),
        "large_angle_mae": float(metrics.large_angle_mae),
        "max_error": float(np.abs(preds - labels).max()),
    }

    print("当前指标:")
    for k, v in current.items():
        print(f"  {k}: {v:.4f}°")

    # 保存基准
    if cli.save_baseline or cli.force_update:
        with open(BASELINE_PATH, "w") as f:
            json.dump(current, f, indent=2)
        print(f"\n基准已保存: {BASELINE_PATH}")
        return 0

    # 对比基准
    if not os.path.exists(BASELINE_PATH):
        print(f"\n未找到基准文件: {BASELINE_PATH}")
        print("请先运行: python check_regression.py --save-baseline")
        return 1

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    print(f"\n基准指标 (来自 {BASELINE_PATH}):")
    for k, v in baseline.items():
        print(f"  {k}: {v:.4f}°")

    # 检查退化（包括分层指标）
    degraded = []
    _check_keys = ("test_mae", "p95", "max_error", "small_angle_mae", "large_angle_mae")
    for k in _check_keys:
        if k not in baseline:
            continue  # 向后兼容旧基线
        old = baseline[k]
        new = current[k]
        if old > 1e-9:
            ratio = new / old
            if ratio > 1.0 + cli.tolerance:
                degraded.append((k, old, new, ratio))

    print(f"\n退化检查 (tolerance={cli.tolerance*100:.0f}%):")
    if degraded:
        for k, old, new, ratio in degraded:
            print(f"  FAIL {k}: {old:.4f}° → {new:.4f}° "
                  f"(恶化 {(ratio-1)*100:.1f}%)")
        print("\n回归测试失败！模型精度退化超过容差。")
        print("可能原因：训练配置变更、数据集变更、代码 bug。")
        print("建议：检查最近的代码改动，恢复到基准版本。")
        return 1
    else:
        print("  PASS 所有指标在容差范围内。")
        # 显示改进
        improved = []
        for k in _check_keys:
            if k not in baseline:
                continue
            old = baseline[k]
            new = current[k]
            if old > 1e-9 and new < old * 0.95:
                improved.append((k, old, new))
        if improved:
            print("\n相比基准有改进:")
            for k, old, new in improved:
                print(f"  {k}: {old:.4f}° → {new:.4f}° "
                      f"(改善 {(1-new/old)*100:.1f}%)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
