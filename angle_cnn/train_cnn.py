#!/usr/bin/env python3
"""
train_cnn.py — MoS₂ moiré 转角 CNN 回归训练（v5：BF16 + compile + EMA / 角度加权）
================================================================================

任务：输入 128×128 MoS₂ moiré 图像，输出转角 θ（度）

v5 新增
------
- ``--ema-decay``：参数滑动平均；验证 / ``best_model.pt`` / 测试使用 EMA 权重（0=关闭）
- ``--angle-loss-weight``：Huber 按归一化标签加权，略缓解高转角端 MAE 偏大（0=普通 Huber）

v4（RTX 5070 Ti Blackwell 优化）
--------------------------------------
- BF16 AMP（--bf16）— Blackwell 原生支持，数值更稳定，速度等同 FP16
- torch.compile 默认启用，可选 mode=reduce-overhead/max-autotune
- 梯度裁剪（--grad-clip 1.0）— 防止 AMP 训练梯度爆炸
- DataLoader prefetch_factor — 减少 GPU 等待数据的空闲时间
- 多骨干选择（--arch resnet18/34/50）— 充分利用 16 GB VRAM
- 训练 batch 末尾 drop_last — AMP 小 batch 稳定性
- 默认 batch_size 512（16 GB VRAM 可轻松承载 128×128×3）

模型：ResNet backbone 改造版
  - conv1: 3×3 stride=2（保留高频 moiré 信息）
  - maxpool 替换为 Identity（避免过度下采样）
  - 输出头：512→256→64→1，双层 Dropout

训练策略
--------
- 损失函数：Huber Loss (δ=0.02)
- 优化器：AdamW + Warmup-Cosine LR
- Mixup 数据增强 (α=0.2)
- 早停：val loss 连续 patience epoch 不下降则停止

运行方式
--------
    python train_cnn.py                           # 默认配置（resnet18, BF16）
    python train_cnn.py --arch resnet34           # 更大模型
    python train_cnn.py --fft-channel             # 含 FFT 额外通道
    python train_cnn.py --batch-size 1024         # 更大 batch（16 GB VRAM 绰绰有余）
    python train_cnn.py --no-compile              # 显式禁用 torch.compile
    # WSL 无 gcc 时脚本会自动跳过 compile；需要 compile 时：sudo apt install build-essential
    python train_cnn.py --eval-only               # 跳过训练，仅测试集 + MC（用 outputs/best_model.pt）
    python train_cnn.py --compile-mode max-autotune  # 最激进编译优化
    python train_cnn.py --no-amp                  # 禁用混合精度
    python train_cnn.py --ema-decay 0.999 --angle-loss-weight 0.25  # 可选：EMA + 高角加权
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from angle_cnn.core.config import THETA_MAX, THETA_MIN
from angle_cnn.core.fonts import setup_matplotlib_cjk_font
from angle_cnn.core.cnn import (
    build_model,
    compute_fft_channel,
    detect_n_channels,
    predict_with_uncertainty,
    warmup_cosine_lr,
)
from angle_cnn.core.io_utils import load_model_checkpoint, load_npz_dataset, state_dict_from_checkpoint
from angle_cnn.core.seed import set_global_seed, worker_init_fn
from angle_cnn.core.augment import get_default_augmentation
from angle_cnn.core.metrics import compute_stratified_metrics, generate_evaluation_report
from angle_cnn.core.train_utils import ema_update, weighted_huber_loss

# ── 默认超参数 ─────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 512   # RTX 5070 Ti 16 GB：128×128×3 图像轻松跑 512+
DEFAULT_NUM_EPOCHS = 200
DEFAULT_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 30
DEFAULT_SEED = 42
DEFAULT_MC_SAMPLES = 30


def _running_in_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _dataloader_num_workers(explicit: int | None = None) -> int:
    """DataLoader worker 数：WSL 默认压低，避免 CPU RAM 不足被 OOM killer 直接 Kill。"""
    if explicit is not None:
        return max(0, int(explicit))
    env = os.environ.get("TORCH_NUM_WORKERS", "").strip()
    if env.isdigit():
        return max(0, int(env))
    cpu = os.cpu_count() or 4
    cap = min(8, max(0, cpu - 1))
    if _running_in_wsl():
        cap = min(cap, 2)
    return cap


def _dataset_ram_gb(data) -> float:
    """估算 npz 中图像数组占用（已解压到内存时）。"""
    total = 0
    for key in ("images_train", "images_val", "images_test"):
        if key in getattr(data, "files", ()):
            total += int(data[key].nbytes)
    return total / (1024.0 ** 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoS₂ moiré CNN 训练 — ResNet 角度回归（多通道 + MC Dropout + 可选 EMA）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="最大训练轮数")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="权重衰减")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="早停 patience")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--data-dir", type=str, default=None, help="数据集目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    parser.add_argument("--no-augment", action="store_true", help="禁用数据增强")
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES,
                        help="MC Dropout 推理采样次数（0 禁用）")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 率")
    parser.add_argument("--huber-delta", type=float, default=0.02,
                        help="Huber loss delta（越小越接近 MAE）")
    parser.add_argument("--fft-channel", action="store_true",
                        help="添加 FFT magnitude 作为额外输入通道")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="Mixup alpha（0 禁用）")
    parser.add_argument("--warmup-epochs", type=int, default=8,
                        help="学习率线性预热轮数")
    parser.add_argument("--no-amp", action="store_true",
                        help="禁用 Mixed Precision（AMP）训练")
    parser.add_argument("--bf16", action="store_true",
                        help="AMP 使用 bfloat16（Blackwell/Ampere 推荐，数值更稳定）")
    parser.add_argument("--no-compile", action="store_true",
                        help="禁用 torch.compile（首次调试或旧版 PyTorch 时使用）")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile 编译模式（max-autotune 最快但编译时间长）")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="梯度裁剪最大范数（0 禁用）")
    parser.add_argument("--prefetch-factor", type=int, default=None,
                        help="DataLoader 预取因子（默认 4；WSL 下自动降为 2 以防内存杀进程）")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader 进程数（默认自动：WSL 最多 2，其它最多 8；也可用环境变量 TORCH_NUM_WORKERS）")
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="骨干网络：resnet18(~11M) | resnet34(~21M) | resnet50(~23M)")
    parser.add_argument("--fp16-data", action="store_true",
                        help="将数据集转为 float16 加载（减少内存和加载时间）")
    parser.add_argument("--no-progress", action="store_true",
                        help="禁用训练/验证时每个 epoch 的 batch 级 tqdm 进度条")
    parser.add_argument("--eval-only", action="store_true",
                        help="跳过训练：从 output-dir/best_model.pt 加载权重，仅跑测试集与 MC（元数据从 ckpt 读取）")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA 滑动平均衰减；0=关闭，建议 0.999–0.9999。开启时验证/保存/测试使用 EMA 权重")
    parser.add_argument("--swa", action="store_true",
                        help="启用 SWA（Stochastic Weight Averaging），从 50%% epoch 开始平均权重，压制 val loss 震荡")
    parser.add_argument("--angle-loss-weight", type=float, default=0.0,
                        help="Huber 按归一化标签加权 w=1+coef·target（高 θ 权重更大）；0=普通 Huber")
    return parser.parse_args()


def _batch_progress_enabled(no_progress_flag: bool) -> bool:
    return (not no_progress_flag) and bool(getattr(sys.stderr, "isatty", lambda: False)()) and tqdm is not None


def _c_compiler_available() -> bool:
    """torch.compile → Inductor/Triton 在首次运行时常需本机 gcc/clang 编译辅助代码。"""
    cc = os.environ.get("CC", "").strip()
    if cc:
        exe = cc.split()[0]
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            return True
        if shutil.which(exe):
            return True
    return bool(shutil.which("gcc") or shutil.which("clang") or shutil.which("cc"))


# ── 设备选择 ──────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("使用 CPU（建议用 GPU 加速）")
    return dev


class MoireDataset(Dataset):
    """MoS₂ moiré 图像数据集（支持单/多通道 + FFT通道 + 在线增强）。

    标签归一化：θ_norm = (θ - THETA_MIN) / (THETA_MAX - THETA_MIN)
    """

    def __init__(self, images, labels, augment=False,
                 theta_min=THETA_MIN, theta_max=THETA_MAX,
                 add_fft_channel=False, fp16=False):
        if images.ndim == 3:
            t = torch.from_numpy(images).unsqueeze(1)
        else:
            t = torch.from_numpy(images)
        self.images = t.half() if fp16 else t
        self.labels = torch.from_numpy(
            (labels - theta_min) / (theta_max - theta_min)
        ).float()
        self.augment = augment
        self.add_fft_channel = add_fft_channel

    @property
    def n_channels(self) -> int:
        n = self.images.shape[1]
        return n + (1 if self.add_fft_channel else 0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].float()
        lbl = self.labels[idx]

        if self.augment:
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            img = torch.rot90(img, k, dims=[1, 2])

            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            img = img * brightness
            mean = img.mean()
            contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            img = (img - mean) * contrast + mean
            img = img.clamp(0, 1)

        # NOTE: FFT 通道在 CPU 上算会很慢。训练脚本会在 GPU 上按需追加 FFT 通道，
        # 这里保留旧行为用于兼容（例如单样本可视化/调试）。
        if self.add_fft_channel:
            img = compute_fft_channel(img)

        return img, lbl


# ── 模型 ──────────────────────────────────────────────────
# ── 训练工具 ──────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, device,
                    mixup_alpha=0.0, scaler=None, use_amp=False,
                    amp_dtype=torch.float16, add_fft_channel: bool = False,
                    grad_clip: float = 1.0, progress: bool = False, desc: str = "train",
                    huber_delta: float = 0.02, angle_loss_weight: float = 0.0,
                    ema_model=None, ema_decay: float = 0.0):
    model.train()
    total_loss = 0.0
    it = loader
    if progress and tqdm is not None:
        it = tqdm(loader, desc=desc, leave=False, file=sys.stderr, dynamic_ncols=True)
    for imgs, lbls in it:
        imgs = imgs.to(device, non_blocking=True)
        if add_fft_channel:
            imgs = compute_fft_channel(imgs)
        lbls = lbls.to(device, non_blocking=True).unsqueeze(1)

        if mixup_alpha > 0:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(imgs.size(0), device=device)
            imgs = lam * imgs + (1 - lam) * imgs[idx]
            lbls = lam * lbls + (1 - lam) * lbls[idx]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            preds = model(imgs)
            loss = weighted_huber_loss(
                preds, lbls, delta=huber_delta, angle_weight=angle_loss_weight,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema_model is not None and ema_decay > 0:
            ema_update(model, ema_model, ema_decay)

        total_loss += loss.item() * len(imgs)
        if progress and tqdm is not None and hasattr(it, "set_postfix"):
            it.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False,
             amp_dtype=torch.float16, add_fft_channel: bool = False,
             progress: bool = False, desc: str = "val",
             huber_delta: float = 0.02, angle_loss_weight: float = 0.0):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    it = loader
    if progress and tqdm is not None:
        it = tqdm(loader, desc=desc, leave=False, file=sys.stderr, dynamic_ncols=True)
    for imgs, lbls in it:
        imgs = imgs.to(device, non_blocking=True)
        if add_fft_channel:
            imgs = compute_fft_channel(imgs)
        lbls = lbls.to(device, non_blocking=True).unsqueeze(1)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            preds = model(imgs)
            loss = weighted_huber_loss(
                preds, lbls, delta=huber_delta, angle_weight=angle_loss_weight,
            )
        total_loss += loss.item() * len(imgs)
        preds_deg = preds.float().squeeze(1).clamp(0, 1) * (THETA_MAX - THETA_MIN) + THETA_MIN
        lbls_deg = lbls.squeeze(1) * (THETA_MAX - THETA_MIN) + THETA_MIN
        total_mae += (preds_deg - lbls_deg).abs().sum().item()
        if progress and tqdm is not None and hasattr(it, "set_postfix"):
            it.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
    n = len(loader.dataset)
    return total_loss / n, total_mae / n


@torch.no_grad()
def predict_testset(model, loader, device, use_amp=False, amp_dtype=torch.float16, add_fft_channel: bool = False):
    """Collect full test predictions/labels in degree for post-hoc reports."""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        if add_fft_channel:
            imgs = compute_fft_channel(imgs)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            preds = model(imgs).float().squeeze(1).clamp(0, 1)
        preds_deg = (preds * (THETA_MAX - THETA_MIN) + THETA_MIN).cpu().numpy()
        lbls_deg = (lbls * (THETA_MAX - THETA_MIN) + THETA_MIN).cpu().numpy()
        all_preds.append(preds_deg)
        all_labels.append(lbls_deg)
    return np.concatenate(all_preds), np.concatenate(all_labels)


def save_log_plot(train_losses, val_losses, val_maes, log_png_path):
    try:
        setup_matplotlib_cjk_font()
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(r"MoS$_2$ moiré CNN 训练曲线", fontsize=13)

        ax1.plot(epochs, train_losses, "b-", lw=1.5, label="train loss")
        ax1.plot(epochs, val_losses, "r-", lw=1.5, label="val loss")
        ax1.set(xlabel="Epoch", ylabel="Huber Loss", title="Loss 曲线")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(epochs, val_maes, "g-", lw=1.5)
        ax2.axhline(0.1, color="red", ls="--", lw=1.2, label="目标 0.1°")
        ax2.set(xlabel="Epoch", ylabel="MAE (°)", title="验证集角度误差")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(log_png_path, dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError as e:
        logging.getLogger(__name__).warning("绘图跳过（matplotlib）: %s", e)
    except OSError as e:
        logging.getLogger(__name__).warning("绘图保存失败: %s", e)


# ── 主程序 ────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    set_global_seed(args.seed)

    data_dir = args.data_dir or os.path.join(SCRIPT_DIR, "..", "data")
    out_dir = args.output_dir or os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, "moire_dataset.npz")
    model_path = os.path.join(out_dir, "best_model.pt")
    log_csv = os.path.join(out_dir, "train_log.csv")
    log_png = os.path.join(out_dir, "train_log.png")

    print("=" * 60)
    print("MoS₂ moiré CNN 训练 v5（BF16 + torch.compile + 可选 EMA / 角度加权）")
    print("=" * 60)

    device = get_device()
    use_cuda = torch.cuda.is_available()
    pin_mem = use_cuda

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024 ** 3
        print(f"GPU 显存: {vram_gb:.1f} GB  |  计算能力: sm_{props.major}{props.minor}")
        bf16_ok = props.major >= 8  # Ampere(sm_80+) 及以上原生 BF16
        if args.bf16 and not bf16_ok:
            print("  警告：当前 GPU 不支持原生 BF16，回退到 FP16")
            args.bf16 = False
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    print(f"\n加载数据集: {dataset_path}")
    data = load_npz_dataset(dataset_path)

    base_n_ch = detect_n_channels(data["images_train"])
    preloaded_ckpt = None
    if args.eval_only:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"--eval-only 需要已有权重: {model_path}\n"
                "  请先完整训练，或用 --output-dir 指向含 best_model.pt 的目录。"
            )
        preloaded_ckpt = load_model_checkpoint(model_path, map_location=device)
        meta = preloaded_ckpt if isinstance(preloaded_ckpt, dict) else {}
        add_fft = bool(meta.get("add_fft_channel", args.fft_channel))
        n_ch = int(meta.get("n_channels", base_n_ch + (1 if add_fft else 0)))
        if meta.get("arch"):
            args.arch = str(meta["arch"])
        if meta.get("dropout") is not None:
            args.dropout = float(meta["dropout"])
        if meta.get("angle_loss_weight") is not None:
            args.angle_loss_weight = float(meta["angle_loss_weight"])
        if meta.get("huber_delta") is not None:
            args.huber_delta = float(meta["huber_delta"])
        if n_ch != base_n_ch + (1 if add_fft else 0):
            add_fft = n_ch > base_n_ch
        print(f"\n--eval-only：将使用 {model_path}")
        print(f"  checkpoint 元数据: arch={args.arch}, n_channels={n_ch}, add_fft={add_fft}, dropout={args.dropout}")
    else:
        add_fft = args.fft_channel
        n_ch = base_n_ch + (1 if add_fft else 0)
    print(f"  检测到 {base_n_ch} 通道数据集" + (f" + FFT通道 = {n_ch} 通道输入" if add_fft else ""))

    angle_loss_w = float(args.angle_loss_weight)
    ema_decay = float(args.ema_decay)
    if ema_decay < 0 or ema_decay >= 1.0:
        raise ValueError("--ema-decay 须满足 0 <= decay < 1；0 表示关闭 EMA")
    ema_active = (not args.eval_only) and ema_decay > 0

    amp_dtype = torch.bfloat16 if (use_cuda and args.bf16) else torch.float16

    fp16_data = args.fp16_data
    # FFT 通道尽量在 GPU 上追加，避免 DataLoader 在 CPU 上做 FFT
    train_ds = MoireDataset(data["images_train"], data["labels_train"],
                            augment=not args.no_augment, add_fft_channel=False,
                            fp16=fp16_data)
    val_ds = MoireDataset(data["images_val"], data["labels_val"],
                          augment=False, add_fft_channel=False, fp16=fp16_data)
    test_ds = MoireDataset(data["images_test"], data["labels_test"],
                           augment=False, add_fft_channel=False, fp16=fp16_data)

    print(f"  train: {len(train_ds)} 样本")
    print(f"  val:   {len(val_ds)} 样本")
    print(f"  test:  {len(test_ds)} 样本")
    try:
        ram_g = _dataset_ram_gb(data)
        print(f"  数据集图像约占用 RAM: {ram_g:.1f} GiB（train+val+test，未含 PyTorch 开销）")
    except Exception:
        ram_g = 0.0
    if _running_in_wsl():
        print(
            "  提示：WSL 默认内存常小于物理机；若训练出现仅有「Killed」无报错，多为系统内存不足。"
            "可：增大 .wslconfig 内存、加 --fp16-data、减小 --batch-size / --num-workers / --prefetch-factor。"
        )

    nw = _dataloader_num_workers(args.num_workers)
    prefetch_eff = args.prefetch_factor if args.prefetch_factor is not None else 4
    if _running_in_wsl() and args.prefetch_factor is None:
        prefetch_eff = min(prefetch_eff, 2)

    print(f"\n配置:")
    print(f"  batch_size:    {args.batch_size}")
    print(f"  epochs:        {args.epochs}")
    print(f"  lr:            {args.lr}")
    print(f"  weight_decay:  {args.weight_decay}")
    print(f"  patience:      {args.patience}")
    print(f"  dropout:       {args.dropout}")
    print(f"  huber_delta:   {args.huber_delta}")
    print(f"  angle_loss_w:  {angle_loss_w}")
    print(f"  ema_decay:     {ema_decay}" + ("（验证/保存/测试用 EMA）" if ema_active else ""))
    print(f"  mc_samples:    {args.mc_samples}")
    print(f"  fft_channel:   {add_fft}")
    print(f"  mixup_alpha:   {args.mixup_alpha}")
    print(f"  warmup_epochs: {args.warmup_epochs}")
    print(f"  augment:       {not args.no_augment}")
    print(f"  n_channels:    {n_ch} ({'基础' + str(base_n_ch) + '+FFT1' if add_fft else str(n_ch)})")

    use_amp = use_cuda and not args.no_amp
    use_compile = use_cuda and not args.no_compile
    if use_compile and not _c_compiler_available():
        print(
            "提示：未检测到 C 编译器（gcc/clang 或有效 CC）。"
            "torch.compile 依赖 Triton 会在运行时编译，需要编译器；已自动跳过 compile。\n"
            "      若需 compile：sudo apt install build-essential   或   export CC=/path/to/gcc"
        )
        use_compile = False
    amp_dtype_name = "bfloat16" if args.bf16 else "float16"
    print(f"  amp ({amp_dtype_name}): {use_amp}")
    print(f"  torch.compile: {use_compile}" + (f" (mode={args.compile_mode})" if use_compile else ""))
    print(f"  cudnn.bench:   {use_cuda}")
    print(f"  arch:          {args.arch}")
    print(f"  grad_clip:     {args.grad_clip}")
    print(f"  num_workers:   {nw}" + ("  (WSL 自动压低)" if _running_in_wsl() and args.num_workers is None else ""))
    print(f"  prefetch:      {prefetch_eff}" + ("  (WSL 默认上限 2)" if _running_in_wsl() and args.prefetch_factor is None else ""))
    show_batch_pb = _batch_progress_enabled(args.no_progress)
    print(f"  batch tqdm:    {'开启' if show_batch_pb else '关闭'}")

    w_init = lambda wid: worker_init_fn(wid, args.seed)  # noqa: E731

    persist = nw > 0
    prefetch = prefetch_eff if nw > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=pin_mem, generator=gen,
        worker_init_fn=w_init if nw > 0 else None,
        persistent_workers=persist,
        prefetch_factor=prefetch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin_mem,
        worker_init_fn=w_init if nw > 0 else None,
        persistent_workers=persist,
        prefetch_factor=prefetch,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin_mem,
        worker_init_fn=w_init if nw > 0 else None,
        persistent_workers=persist,
        prefetch_factor=prefetch,
    )

    model = build_model(n_channels=n_ch, dropout=args.dropout, arch=args.arch).to(device)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)

    ema_model = None

    if use_compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"  torch.compile 已启用（mode={args.compile_mode}）")
        except Exception as e:
            print(f"  torch.compile 失败（需要 PyTorch 2.0+）: {e}")

    # BF16 不需要 GradScaler（动态范围足够），FP16 才需要
    need_scaler = use_amp and not args.bf16
    scaler = torch.amp.GradScaler("cuda", enabled=need_scaler) if need_scaler else None

    start_epoch = 1
    best_val_loss = float("inf")
    if preloaded_ckpt is not None:
        _tgt = model._orig_mod if hasattr(model, "_orig_mod") else model
        _tgt.load_state_dict(state_dict_from_checkpoint(preloaded_ckpt))
    elif args.resume:
        print(f"\n从 checkpoint 恢复: {args.resume}")
        ckpt = load_model_checkpoint(args.resume, map_location=device)
        _tgt = model._orig_mod if hasattr(model, "_orig_mod") else model
        _tgt.load_state_dict(state_dict_from_checkpoint(ckpt))
        if isinstance(ckpt, dict):
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            if "val_loss" in ckpt:
                best_val_loss = ckpt["val_loss"]

    if not args.eval_only and ema_decay > 0:
        ema_model = build_model(n_channels=n_ch, dropout=args.dropout, arch=args.arch).to(device)
        if use_cuda:
            ema_model = ema_model.to(memory_format=torch.channels_last)
        # model may be wrapped by torch.compile (_orig_mod. prefix);
        # unwrap to get clean state_dict for the raw ema_model.
        _src = model._orig_mod if hasattr(model, "_orig_mod") else model
        ema_model.load_state_dict(_src.state_dict())

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params/1e6:.2f} M  (arch={args.arch})")
    print(f"DataLoader: num_workers={nw}, pin_memory={pin_mem}, "
          f"persistent_workers={persist}, prefetch_factor={prefetch}, drop_last=True(train)")
    if use_amp:
        print(f"Mixed Precision: AMP ({amp_dtype_name}) 已启用"
              + ("  [无 GradScaler]" if not need_scaler else "  [GradScaler 启用]"))

    train_losses, val_losses, val_maes = [], [], []

    if not args.eval_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: warmup_cosine_lr(e, args.warmup_epochs, args.epochs),
        )

        # SWA：从训练后半段开始对权重做滑动平均，压制 val loss 震荡
        swa_model = None
        swa_start = args.epochs // 2
        if args.swa:
            from torch.optim.swa_utils import AveragedModel, SWALR
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.1)

        print(f"\n开始训练（最多 {args.epochs} epoch，早停 patience={args.patience})"
              + (f"，SWA 从 epoch {swa_start} 开始" if args.swa else ""))
        print(f'{"Epoch":>6}  {"Train Loss":>11}  {"Val Loss":>10}  '
              f'{"Val MAE":>9}  {"LR":>9}  {"时间":>6}')
        print("─" * 62)

        patience_cnt = 0

        if start_epoch == 1:
            with open(log_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "val_mae_deg", "lr"])

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, device,
                mixup_alpha=args.mixup_alpha,
                scaler=scaler, use_amp=use_amp, amp_dtype=amp_dtype,
                add_fft_channel=add_fft, grad_clip=args.grad_clip,
                progress=show_batch_pb, desc=f"ep{epoch} train",
                huber_delta=args.huber_delta, angle_loss_weight=angle_loss_w,
                ema_model=ema_model, ema_decay=ema_decay,
            )
            val_net = ema_model if (ema_model is not None and ema_decay > 0) else model
            val_loss, val_mae = evaluate(
                val_net, val_loader, device,
                use_amp=use_amp, amp_dtype=amp_dtype, add_fft_channel=add_fft,
                progress=show_batch_pb, desc=f"ep{epoch} val",
                huber_delta=args.huber_delta, angle_loss_weight=angle_loss_w,
            )
            scheduler.step()
            if args.swa and swa_model is not None and epoch >= swa_start:
                swa_model.update_parameters(model)
                if swa_scheduler is not None:
                    swa_scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            dt = time.time() - t0

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_maes.append(val_mae)

            flag = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0
                _save_src = ema_model if (ema_model is not None and ema_decay > 0) else (model._orig_mod if hasattr(model, "_orig_mod") else model)
                save_sd = _save_src.state_dict()
                torch.save(
                    {
                        "model_state_dict": save_sd,
                        "model_version": "v5",
                        "arch": args.arch,
                        "theta_min": THETA_MIN,
                        "theta_max": THETA_MAX,
                        "n_channels": n_ch,
                        "base_n_channels": base_n_ch,
                        "add_fft_channel": add_fft,
                        "dropout": args.dropout,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_mae_deg": val_mae,
                        "ema_decay": ema_decay,
                        "angle_loss_weight": angle_loss_w,
                        "huber_delta": float(args.huber_delta),
                    },
                    model_path,
                )
                flag = " ✓"
            else:
                patience_cnt += 1

            print(
                f"{epoch:>6}  {train_loss:>11.6f}  {val_loss:>10.6f}  "
                f"{val_mae:>8.4f}°  {lr:>9.2e}  {dt:>5.1f}s{flag}"
            )

            if epoch % 10 == 0:
                save_log_plot(train_losses, val_losses, val_maes, log_png)

            with open(log_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, val_mae, lr])

            if patience_cnt >= args.patience:
                print(f"\n早停触发（{args.patience} epoch val loss 未改善）")
                break

        # SWA 最终化：用训练集更新 BN 统计量，然后评估
        if args.swa and swa_model is not None:
            import torch.optim.swa_utils as swa_utils
            print(f"\nSWA 最终化：用训练集更新 BN 统计量...")
            swa_utils.update_bn(train_loader, swa_model, device=device)
            swa_val_loss, swa_val_mae = evaluate(
                swa_model, val_loader, device,
                use_amp=use_amp, amp_dtype=amp_dtype, add_fft_channel=add_fft,
                huber_delta=args.huber_delta, angle_loss_weight=angle_loss_w,
            )
            print(f"  SWA val loss={swa_val_loss:.6f}  val MAE={swa_val_mae:.4f}°")
            if swa_val_loss < best_val_loss:
                print(f"  SWA 模型优于 best checkpoint，保存 SWA 权重")
                best_val_loss = swa_val_loss
                _save_src = swa_model.module if hasattr(swa_model, "module") else swa_model
                torch.save(
                    {
                        "model_state_dict": _save_src.state_dict(),
                        "model_version": "v5-swa",
                        "arch": args.arch,
                        "theta_min": THETA_MIN,
                        "theta_max": THETA_MAX,
                        "n_channels": n_ch,
                        "base_n_channels": base_n_ch,
                        "add_fft_channel": add_fft,
                        "dropout": args.dropout,
                        "epoch": epoch,
                        "val_loss": swa_val_loss,
                        "val_mae_deg": swa_val_mae,
                        "ema_decay": ema_decay,
                        "swa": True,
                        "angle_loss_weight": angle_loss_w,
                        "huber_delta": float(args.huber_delta),
                    },
                    model_path,
                )
    else:
        print("\n--eval-only：跳过训练，进入测试集与 MC 评估")

    # ── 测试集评估 ──
    if not args.eval_only:
        print(f"\n加载最佳模型: {model_path}")
        ckpt = load_model_checkpoint(model_path, map_location=device)
        _tgt = model._orig_mod if hasattr(model, "_orig_mod") else model
        _tgt.load_state_dict(state_dict_from_checkpoint(ckpt))
    else:
        print(f"\n使用已加载的 best_model.pt（--eval-only）")

    test_loss, test_mae = evaluate(
        model, test_loader, device,
        use_amp=use_amp, amp_dtype=amp_dtype, add_fft_channel=add_fft,
        huber_delta=args.huber_delta, angle_loss_weight=angle_loss_w,
    )
    print(f"测试集 MAE:  {test_mae:.4f}°")
    print(f"测试集 Loss: {test_loss:.6f}")

    # ── MC Dropout 不确定性评估 ──
    if args.mc_samples > 0:
        print(f"\nMC Dropout 不确定性评估（{args.mc_samples} 次采样）...")
        all_means, all_stds, all_labels = [], [], []
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            if add_fft:
                imgs = compute_fft_channel(imgs)
            mean_deg, std_deg = predict_with_uncertainty(model, imgs, n_samples=args.mc_samples)
            all_means.append(mean_deg)
            all_stds.append(std_deg)
            lbls_deg = lbls.numpy() * (THETA_MAX - THETA_MIN) + THETA_MIN
            all_labels.append(lbls_deg)
        all_means = np.concatenate(all_means)
        all_stds = np.concatenate(all_stds)
        all_labels = np.concatenate(all_labels)
        mc_mae = np.abs(all_means - all_labels).mean()
        mean_unc = all_stds.mean()
        print(f"  MC MAE:       {mc_mae:.4f}°")
        print(f"  平均不确定性: ±{mean_unc:.4f}°")
        print(f"  不确定性范围: [{all_stds.min():.4f}°, {all_stds.max():.4f}°]")

    # ── 分层评估与校准报告（测试集）──
    print("\n生成分层评估报告...")
    if args.mc_samples > 0:
        report_preds = all_means
        report_labels = all_labels
        report_uncs = all_stds
    else:
        report_preds, report_labels = predict_testset(
            model, test_loader, device, use_amp=use_amp, amp_dtype=amp_dtype, add_fft_channel=add_fft
        )
        report_uncs = None

    report = generate_evaluation_report(
        report_preds,
        report_labels,
        output_dir=out_dir,
        method_name="CNN",
        uncertainties=report_uncs,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
    )
    print(
        "  分层指标: "
        f"P95={report.get('p95', float('nan')):.4f}°, "
        f"small<1.5° MAE={report.get('small_angle_mae', float('nan')):.4f}°, "
        f"large>3.5° MAE={report.get('large_angle_mae', float('nan')):.4f}°"
    )

    # ── 论文表格友好：单行 CSV 摘要 ──
    summary_csv = os.path.join(out_dir, "train_test_summary.csv")
    summary_row = {
        "n_samples": int(len(report_labels)),
        "model_arch": args.arch,
        "n_channels": int(n_ch),
        "add_fft_channel": bool(add_fft),
        "test_mae_deg": float(test_mae),
        "test_loss": float(test_loss),
        "p95_deg": float(report.get("p95", np.nan)),
        "p99_deg": float(report.get("p99", np.nan)),
        "small_lt1p5_mae_deg": float(report.get("small_angle_mae", np.nan)),
        "large_gt3p5_mae_deg": float(report.get("large_angle_mae", np.nan)),
    }
    if args.mc_samples > 0:
        summary_row.update({
            "mc_samples": int(args.mc_samples),
            "mc_mae_deg": float(mc_mae),
            "mean_unc_deg": float(mean_unc),
        })
        cal = report.get("calibration", {})
        if isinstance(cal, dict):
            summary_row.update({
                "within_1sigma_pct": float(cal.get("within_1sigma_pct", np.nan)),
                "within_2sigma_pct": float(cal.get("within_2sigma_pct", np.nan)),
                "within_3sigma_pct": float(cal.get("within_3sigma_pct", np.nan)),
                "error_unc_corr": float(cal.get("error_uncertainty_correlation", np.nan)),
                "calibration_mse": float(cal.get("calibration_mse", np.nan)),
            })

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    print(f"  测试摘要CSV: {summary_csv}")

    if train_losses:
        save_log_plot(train_losses, val_losses, val_maes, log_png)
        print(f"\n训练曲线: {log_png}")
    elif args.eval_only:
        print("\n（--eval-only 未产生新训练曲线）")
    print(f"数值日志: {log_csv}")
    print(f"模型权重: {model_path}")

    print("\n" + "=" * 60)
    print(f"CNN 测试集 MAE:  {test_mae:.4f}°")
    if args.mc_samples > 0:
        print(f"MC Dropout 不确定性: ±{mean_unc:.4f}°")
    print(f"通道数: {n_ch} ({'含FFT' if add_fft else '无FFT'})")
    print(f"架构: conv1=3x3s2, no maxpool, linear output")
    ema_note = f", EMA(decay={ema_decay})" if ema_active else ""
    ang_note = f", angle-weight={angle_loss_w}" if angle_loss_w > 0 else ""
    print(
        f"训练策略: Huber(δ={args.huber_delta}){ang_note}, Mixup(α={args.mixup_alpha}), "
        f"Warmup({args.warmup_epochs}ep)+Cosine{ema_note}"
        + (", SWA" if args.swa else "")
    )
    print(f"加速: AMP({amp_dtype_name})={'启用' if use_amp else '禁用'}, "
          f"compile={'启用' if use_compile else '禁用'}, batch_size={args.batch_size}, "
          f"grad_clip={args.grad_clip}")
    print("下一步：python eval_compare.py（CNN vs FFT 对比）")
    print("=" * 60)


if __name__ == "__main__":
    main()
