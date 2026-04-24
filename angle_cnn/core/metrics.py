#!/usr/bin/env python3
"""
metrics.py — MoS₂ moiré CNN 分层评估指标模块
==============================================

提供全面的评估指标，包括：
- 分层 MAE（按角度区间统计）
- 百分位误差（P50, P90, P95, P99）
- 误差分布可视化
- 小角度性能专项评估
- 误差-不确定性校准分析

设计理念
--------
传统 MAE 对异常值不敏感，实际应用中大误差更致命。
本模块提供更全面的评估，帮助定位模型的弱点。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StratifiedMetrics:
    """分层评估指标结果。

    Attributes
    ----------
    overall_mae : float
        整体 MAE
    median_error : float
        中位误差
    percentile_errors : dict
        百分位误差 {50: x, 90: y, 95: z, 99: w}
    angle_bin_mae : dict
        按角度区间分层的 MAE {bin_center: mae}
    small_angle_mae : float
        小角度区间（θ < 1.5°）的 MAE
    large_angle_mae : float
        大角度区间（θ > 3.5°）的 MAE
    error_std : float
        误差标准差
    max_error : float
        最大误差
    failure_rate : float
        失败率（NaN 预测比例）
    """

    overall_mae: float = 0.0
    median_error: float = 0.0
    percentile_errors: Dict[int, float] = field(default_factory=dict)
    angle_bin_mae: Dict[float, float] = field(default_factory=dict)
    small_angle_mae: float = 0.0
    large_angle_mae: float = 0.0
    error_std: float = 0.0
    max_error: float = 0.0
    failure_rate: float = 0.0
    n_samples: int = 0
    n_valid: int = 0

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "overall_mae": self.overall_mae,
            "median_error": self.median_error,
            "p50": self.percentile_errors.get(50, 0.0),
            "p90": self.percentile_errors.get(90, 0.0),
            "p95": self.percentile_errors.get(95, 0.0),
            "p99": self.percentile_errors.get(99, 0.0),
            "small_angle_mae": self.small_angle_mae,
            "large_angle_mae": self.large_angle_mae,
            "error_std": self.error_std,
            "max_error": self.max_error,
            "failure_rate": self.failure_rate,
            "n_samples": self.n_samples,
            "n_valid": self.n_valid,
        }

    def summary(self) -> str:
        """生成文本摘要。"""
        lines = [
            f"整体 MAE:     {self.overall_mae:.4f}°",
            f"中位误差:     {self.median_error:.4f}°",
            f"P90 误差:     {self.percentile_errors.get(90, 0.0):.4f}°",
            f"P95 误差:     {self.percentile_errors.get(95, 0.0):.4f}°",
            f"P99 误差:     {self.percentile_errors.get(99, 0.0):.4f}°",
            f"小角度 MAE:   {self.small_angle_mae:.4f}° (θ < 1.5°)",
            f"大角度 MAE:   {self.large_angle_mae:.4f}° (θ > 3.5°)",
            f"误差 std:     {self.error_std:.4f}°",
            f"最大误差:     {self.max_error:.4f}°",
            f"失败率:       {self.failure_rate:.2f}%",
        ]
        return "\n".join(lines)


def compute_stratified_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    theta_min: float = 0.5,
    theta_max: float = 5.0,
    n_angle_bins: int = 10,
    small_angle_threshold: float = 1.5,
    large_angle_threshold: float = 3.5,
    percentiles: tuple[int, ...] = (50, 90, 95, 99),
) -> StratifiedMetrics:
    """计算分层评估指标。

    Parameters
    ----------
    predictions : np.ndarray
        预测角度数组，形状 (N,)
    labels : np.ndarray
        真实角度数组，形状 (N,)
    theta_min : float
        最小角度
    theta_max : float
        最大角度
    n_angle_bins : int
        角度区间数量
    small_angle_threshold : float
        小角度阈值
    large_angle_threshold : float
        大角度阈值
    percentiles : list
        需要计算的百分位

    Returns
    -------
    StratifiedMetrics
        分层评估指标结果
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    n_samples = len(labels)
    valid_mask = ~np.isnan(predictions) & ~np.isnan(labels)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        return StratifiedMetrics(
            failure_rate=100.0,
            n_samples=n_samples,
            n_valid=0,
        )

    valid_preds = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    errors = np.abs(valid_preds - valid_labels)

    # 整体指标
    overall_mae = errors.mean()
    median_error = np.median(errors)
    error_std = errors.std()
    max_error = errors.max()
    failure_rate = (n_samples - n_valid) / n_samples * 100

    # 百分位误差
    percentile_errors = {p: np.percentile(errors, p) for p in percentiles}

    # 角度区间分层 MAE
    bin_edges = np.linspace(theta_min, theta_max, n_angle_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    angle_bin_mae = {}

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (valid_labels >= lo) & (valid_labels < hi)
        if i == n_angle_bins - 1:  # 最后一个区间包含上界
            mask = (valid_labels >= lo) & (valid_labels <= hi)
        if mask.sum() > 0:
            angle_bin_mae[bin_centers[i]] = errors[mask].mean()
        else:
            angle_bin_mae[bin_centers[i]] = np.nan

    # 小角度和大角度专项评估
    small_mask = valid_labels < small_angle_threshold
    large_mask = valid_labels > large_angle_threshold

    small_angle_mae = errors[small_mask].mean() if small_mask.sum() > 0 else np.nan
    large_angle_mae = errors[large_mask].mean() if large_mask.sum() > 0 else np.nan

    return StratifiedMetrics(
        overall_mae=overall_mae,
        median_error=median_error,
        percentile_errors=percentile_errors,
        angle_bin_mae=angle_bin_mae,
        small_angle_mae=small_angle_mae,
        large_angle_mae=large_angle_mae,
        error_std=error_std,
        max_error=max_error,
        failure_rate=failure_rate,
        n_samples=n_samples,
        n_valid=n_valid,
    )


def compute_calibration_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """计算不确定性校准指标。

    评估预测不确定性与实际误差的一致性。
    理想情况下，68% 的误差应在 ±1σ 内，95% 应在 ±2σ 内。

    Parameters
    ----------
    predictions : np.ndarray
        预测角度
    labels : np.ndarray
        真实角度
    uncertainties : np.ndarray
        预测不确定性（标准差）
    n_bins : int
        校准曲线的分箱数

    Returns
    -------
    dict
        校准指标
    """
    valid = ~np.isnan(predictions) & ~np.isnan(labels) & ~np.isnan(uncertainties)
    preds = predictions[valid]
    labs = labels[valid]
    uncs = uncertainties[valid]

    errors = np.abs(preds - labs)

    # 覆盖率：误差在 nσ 内的比例
    within_1sigma = (errors <= uncs).mean()
    within_2sigma = (errors <= 2 * uncs).mean()
    within_3sigma = (errors <= 3 * uncs).mean()

    # 校准误差：预测误差与不确定性的相关性
    # 理想情况下，高不确定性应对应大误差
    correlation = np.corrcoef(errors, uncs)[0, 1] if len(errors) > 1 else 0.0

    # 分位数校准曲线
    unc_ranks = np.argsort(uncs)
    n = len(uncs)
    bin_size = n // n_bins

    calibration_curve = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        bin_idx = unc_ranks[start:end]
        mean_unc = uncs[bin_idx].mean()
        mean_err = errors[bin_idx].mean()
        calibration_curve.append((mean_unc, mean_err))

    # 校准均方误差
    cal_mse = np.mean([(u - e) ** 2 for u, e in calibration_curve])

    return {
        "within_1sigma_pct": within_1sigma * 100,
        "within_2sigma_pct": within_2sigma * 100,
        "within_3sigma_pct": within_3sigma * 100,
        "error_uncertainty_correlation": correlation,
        "calibration_mse": cal_mse,
        "expected_1sigma": 68.27,  # 理论值
        "expected_2sigma": 95.45,
    }


def plot_error_distribution(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    uncertainties: Optional[np.ndarray] = None,
    theta_min: float = 0.5,
    theta_max: float = 5.0,
    method_name: str = "CNN",
):
    """绘制误差分布分析图。

    Parameters
    ----------
    predictions : np.ndarray
        预测角度
    labels : np.ndarray
        真实角度
    save_path : str
        保存路径
    uncertainties : np.ndarray, optional
        不确定性
    theta_min : float
        最小角度
    theta_max : float
        最大角度
    method_name : str
        方法名称
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 不可用，跳过绘图")
        return

    valid = ~np.isnan(predictions) & ~np.isnan(labels)
    preds = predictions[valid]
    labs = labels[valid]
    errors = np.abs(preds - labs)

    if uncertainties is not None:
        uncs = uncertainties[valid]
    else:
        uncs = None

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{method_name} 误差分布分析", fontsize=14)

    # 1. 误差直方图 + 百分位线
    ax = axes[0, 0]
    ax.hist(errors, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    for p, c in [(90, "orange"), (95, "red"), (99, "darkred")]:
        val = np.percentile(errors, p)
        ax.axvline(val, color=c, ls="--", lw=1.5, label=f"P{p}={val:.3f}°")
    ax.axvline(errors.mean(), color="green", ls="-", lw=2, label=f"MAE={errors.mean():.3f}°")
    ax.set(xlabel="|角度误差| (°)", ylabel="样本数", title="误差分布")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 2. 角度-误差散点图
    ax = axes[0, 1]
    ax.scatter(labs, errors, alpha=0.3, s=10, c="steelblue")

    # 分层 MAE 曲线
    bin_edges = np.linspace(theta_min, theta_max, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_mae = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (labs >= lo) & (labs < hi)
        bin_mae.append(errors[mask].mean() if mask.sum() > 0 else np.nan)
    ax.plot(bin_centers, bin_mae, "ro-", lw=2, ms=6, label="区间 MAE")

    ax.set(xlabel="真实角度 θ (°)", ylabel="|角度误差| (°)", title="角度 vs 误差")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3. 预测 vs 真实散点图
    ax = axes[1, 0]
    ax.scatter(labs, preds, alpha=0.3, s=10, c="steelblue")
    theta_line = np.linspace(theta_min, theta_max, 100)
    ax.plot(theta_line, theta_line, "r--", lw=1.5, label="理想 y=x")
    ax.set(xlabel="真实角度 θ (°)", ylabel="预测角度 (°)", title="预测 vs 真实")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(theta_min - 0.2, theta_max + 0.2)
    ax.set_ylim(theta_min - 0.2, theta_max + 0.2)

    # 4. 误差-不确定性校准（如果有不确定性）
    ax = axes[1, 1]
    if uncs is not None:
        ax.scatter(uncs, errors, alpha=0.3, s=10, c="steelblue")
        max_val = max(uncs.max(), errors.max())
        ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="理想校准")
        ax.set(xlabel="预测不确定性 σ (°)", ylabel="实际误差 (°)", title="不确定性校准")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 标注校准指标
        within_1sigma = (errors <= uncs).mean() * 100
        ax.text(
            0.05, 0.95,
            f"1σ 覆盖率: {within_1sigma:.1f}%\n理想: 68.3%",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        # 如果没有不确定性，绘制误差 CDF
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cdf, "steelblue", lw=2)
        for p, c in [(90, "orange"), (95, "red"), (99, "darkred")]:
            val = np.percentile(errors, p)
            ax.axvline(val, color=c, ls="--", lw=1.5, label=f"P{p}={val:.3f}°")
        ax.set(xlabel="|角度误差| (°)", ylabel="累积概率", title="误差累积分布函数 (CDF)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("已保存: %s", save_path)


def plot_comparison(
    predictions_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    save_path: str,
    theta_min: float = 0.5,
    theta_max: float = 5.0,
):
    """绘制多方法对比图。

    Parameters
    ----------
    predictions_dict : dict
        {方法名: 预测数组}
    labels : np.ndarray
        真实角度
    save_path : str
        保存路径
    theta_min : float
        最小角度
    theta_max : float
        最大角度
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 不可用，跳过绘图")
        return

    n_methods = len(predictions_dict)
    colors = ["steelblue", "darkorange", "mediumseagreen", "coral", "purple"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("方法对比分析", fontsize=14)

    # 1. MAE 对比柱状图
    ax = axes[0]
    maes = []
    names = []
    for name, preds in predictions_dict.items():
        valid = ~np.isnan(preds) & ~np.isnan(labels)
        mae = np.abs(preds[valid] - labels[valid]).mean()
        maes.append(mae)
        names.append(name)
    bars = ax.bar(names, maes, color=colors[:n_methods], alpha=0.8)
    ax.set(ylabel="MAE (°)", title="整体 MAE 对比")
    ax.grid(axis="y", alpha=0.3)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mae:.3f}°",
                ha="center", va="bottom", fontsize=10)

    # 2. 角度区间 MAE 对比
    ax = axes[1]
    bin_edges = np.linspace(theta_min, theta_max, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i, (name, preds) in enumerate(predictions_dict.items()):
        valid = ~np.isnan(preds) & ~np.isnan(labels)
        valid_preds = preds[valid]
        valid_labels = labels[valid]
        errors = np.abs(valid_preds - valid_labels)

        bin_mae = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (valid_labels >= lo) & (valid_labels < hi)
            bin_mae.append(errors[mask].mean() if mask.sum() > 0 else np.nan)

        ax.plot(bin_centers, bin_mae, "o-", color=colors[i], ms=6, lw=2, label=name)

    ax.axhline(0.1, color="red", ls="--", lw=1.2, label="0.1° 基准线")
    ax.set(xlabel="真实角度 θ (°)", ylabel="MAE (°)", title="各角度区间 MAE")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3. 误差分布对比
    ax = axes[2]
    for i, (name, preds) in enumerate(predictions_dict.items()):
        valid = ~np.isnan(preds) & ~np.isnan(labels)
        errors = np.abs(preds[valid] - labels[valid])
        ax.hist(errors, bins=30, alpha=0.5, color=colors[i], label=f"{name} (MAE={errors.mean():.3f}°)")

    ax.set(xlabel="|角度误差| (°)", ylabel="样本数", title="误差分布对比")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("已保存: %s", save_path)


def generate_evaluation_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    method_name: str = "CNN",
    uncertainties: Optional[np.ndarray] = None,
    theta_min: float = 0.5,
    theta_max: float = 5.0,
) -> Dict:
    """生成完整的评估报告。

    Parameters
    ----------
    predictions : np.ndarray
        预测角度
    labels : np.ndarray
        真实角度
    output_dir : str
        输出目录
    method_name : str
        方法名称
    uncertainties : np.ndarray, optional
        不确定性
    theta_min : float
        最小角度
    theta_max : float
        最大角度

    Returns
    -------
    dict
        评估结果字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # 计算分层指标
    metrics = compute_stratified_metrics(
        predictions, labels, theta_min, theta_max
    )

    # 计算校准指标（如果有不确定性）
    calibration = None
    if uncertainties is not None:
        calibration = compute_calibration_metrics(predictions, labels, uncertainties)

    # 绘制误差分布图
    plot_path = os.path.join(output_dir, f"{method_name.lower()}_error_analysis.png")
    plot_error_distribution(
        predictions, labels, plot_path, uncertainties,
        theta_min, theta_max, method_name
    )

    # 保存文本报告
    report_path = os.path.join(output_dir, f"{method_name.lower()}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{method_name} 评估报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(metrics.summary())
        f.write("\n\n")
        if calibration:
            f.write("不确定性校准:\n")
            f.write(f"  1σ 覆盖率: {calibration['within_1sigma_pct']:.1f}% (理想 68.3%)\n")
            f.write(f"  2σ 覆盖率: {calibration['within_2sigma_pct']:.1f}% (理想 95.4%)\n")
            f.write(f"  误差-不确定性相关系数: {calibration['error_uncertainty_correlation']:.3f}\n")

    logger.info("报告已保存: %s", report_path)

    result = metrics.to_dict()
    if calibration:
        result["calibration"] = calibration

    return result