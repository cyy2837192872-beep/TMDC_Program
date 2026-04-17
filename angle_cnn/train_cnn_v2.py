#!/usr/bin/env python3
"""
train_cnn_v2.py — MoS₂ moiré CNN 增强版训练脚本
================================================

基于 v4 版本，集成以下改进：
1. 多尺度数据增强（提升尺度不变性）
2. 分层评估指标（P90/P95/P99、角度区间 MAE）
3. 频域双流架构（可选）
4. 不确定性校准评估

新增命令行参数
--------------
--scale-augment : 启用多尺度增强
--scale-range : 缩放范围，默认 (0.8, 1.2)
--dual-stream : 使用频域双流架构
--lightweight : 使用轻量级双流架构
--fusion-type : 双流融合类型 (concat/attention)

运行示例
--------
    # 标准训练 + 多尺度增强
    python train_cnn_v2.py --scale-augment

    # 双流架构 + 注意力融合
    python train_cnn_v2.py --dual-stream --fusion-type attention

    # 轻量级双流 + 多尺度增强
    python train_cnn_v2.py --dual-stream --lightweight --scale-augment

    # 完整评估报告
    python train_cnn_v2.py --mc-samples 50 --scale-augment
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.fonts import setup_matplotlib_cjk_font  # noqa: E402
from core.cnn import (  # noqa: E402
    THETA_MAX,
    THETA_MIN,
    build_model,
    compute_fft_channel,
    detect_n_channels,
    predict_with_uncertainty,
    warmup_cosine_lr,
)
from core.io_utils import load_model_checkpoint, load_npz_dataset, state_dict_from_checkpoint  # noqa: E402
from core.seed import set_global_seed, worker_init_fn  # noqa: E402
from core.augment import get_default_augmentation, RandomZoom, RandomFlip, RandomRotation90, Compose  # noqa: E402
from core.metrics import compute_stratified_metrics, generate_evaluation_report, compute_calibration_metrics  # noqa: E402
from core.dual_stream import build_dual_stream_model  # noqa: E402

# ── 默认超参数 ─────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_EPOCHS = 200
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 30
DEFAULT_SEED = 42
DEFAULT_MC_SAMPLES = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoS₂ moiré CNN 增强版训练 — 支持多尺度增强、双流架构",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础参数
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--huber-delta", type=float, default=0.02)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--fp16-data", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--eval-only", action="store_true")

    # v2 新增参数
    parser.add_argument("--scale-augment", action="store_true",
                        help="启用多尺度数据增强")
    parser.add_argument("--scale-range", type=float, nargs=2, default=[0.8, 1.2],
                        help="多尺度增强的缩放范围")
    parser.add_argument("--dual-stream", action="store_true",
                        help="使用频域双流架构")
    parser.add_argument("--lightweight", action="store_true",
                        help="使用轻量级双流架构")
    parser.add_argument("--fusion-type", type=str, default="concat",
                        choices=["concat", "attention"],
                        help="双流融合类型")
    parser.add_argument("--fft-channel", action="store_true",
                        help="添加 FFT 额外通道（仅标准架构）")

    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("使用 CPU（建议用 GPU 加速）")
    return dev


class MoireDatasetV2(Dataset):
    """增强版数据集，支持多尺度增强。"""

    def __init__(self, images, labels, augment=False,
                 theta_min=THETA_MIN, theta_max=THETA_MAX,
                 fp16=False, scale_augment=False, scale_range=(0.8, 1.2)):
        if images.ndim == 3:
            t = torch.from_numpy(images).unsqueeze(1)
        else:
            t = torch.from_numpy(images)
        self.images = t.half() if fp16 else t
        self.labels = torch.from_numpy(
            (labels - theta_min) / (theta_max - theta_min)
        ).float()
        self.augment = augment
        self.scale_augment = scale_augment
        self.scale_range = scale_range

        if scale_augment and augment:
            self.augmentor = Compose([
                RandomFlip(),
                RandomRotation90(),
                RandomZoom(zoom_range=scale_range, target_size=128),
            ])
        elif augment:
            self.augmentor = None  # 使用内置简单增强
        else:
            self.augmentor = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].float()
        lbl = self.labels[idx]

        if self.augmentor is not None:
            rng = np.random.default_rng()
            img = self.augmentor(img, rng)
        elif self.augment:
            # 简单增强（原版行为）
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

        return img, lbl


def train_one_epoch(model, loader, criterion, optimizer, device,
                    mixup_alpha=0.0, scaler=None, use_amp=False,
                    amp_dtype=torch.float16, add_fft_channel: bool = False,
                    grad_clip: float = 1.0, progress: bool = False, desc: str = "train"):
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
            loss = criterion(preds, lbls)

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

        total_loss += loss.item() * len(imgs)
        if progress and tqdm is not None and hasattr(it, "set_postfix"):
            it.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False,
             amp_dtype=torch.float16, add_fft_channel: bool = False,
             progress: bool = False, desc: str = "val"):
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
            loss = criterion(preds, lbls)
        total_loss += loss.item() * len(imgs)
        preds_deg = preds.float().squeeze(1).clamp(0, 1) * (THETA_MAX - THETA_MIN) + THETA_MIN
        lbls_deg = lbls.squeeze(1) * (THETA_MAX - THETA_MIN) + THETA_MIN
        total_mae += (preds_deg - lbls_deg).abs().sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_mae / n


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    set_global_seed(args.seed)

    data_dir = args.data_dir or os.path.join(SCRIPT_DIR, "..", "data")
    out_dir = args.output_dir or os.path.join(SCRIPT_DIR, "outputs_v2")
    os.makedirs(out_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, "moire_dataset.npz")
    model_path = os.path.join(out_dir, "best_model.pt")
    log_csv = os.path.join(out_dir, "train_log.csv")

    print("=" * 60)
    print("MoS₂ moiré CNN 增强版训练 v2")
    print("=" * 60)

    device = get_device()
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024 ** 3
        print(f"GPU 显存: {vram_gb:.1f} GB  |  计算能力: sm_{props.major}{props.minor}")
        bf16_ok = props.major >= 8
        if args.bf16 and not bf16_ok:
            print("  警告：当前 GPU 不支持原生 BF16，回退到 FP16")
            args.bf16 = False

    print(f"\n加载数据集: {dataset_path}")
    data = load_npz_dataset(dataset_path)
    base_n_ch = detect_n_channels(data["images_train"])

    # 架构选择
    use_dual_stream = args.dual_stream
    add_fft = args.fft_channel and not use_dual_stream
    n_ch = base_n_ch + (1 if add_fft else 0)

    amp_dtype = torch.bfloat16 if (use_cuda and args.bf16) else torch.float16
    amp_dtype_name = "bfloat16" if args.bf16 else "float16"
    use_amp = use_cuda and not args.no_amp

    # 数据集
    train_ds = MoireDatasetV2(
        data["images_train"], data["labels_train"],
        augment=not args.no_augment,
        fp16=args.fp16_data,
        scale_augment=args.scale_augment,
        scale_range=tuple(args.scale_range),
    )
    val_ds = MoireDatasetV2(
        data["images_val"], data["labels_val"],
        augment=False, fp16=args.fp16_data,
    )
    test_ds = MoireDatasetV2(
        data["images_test"], data["labels_test"],
        augment=False, fp16=args.fp16_data,
    )

    print(f"  train: {len(train_ds)} 样本")
    print(f"  val:   {len(val_ds)} 样本")
    print(f"  test:  {len(test_ds)} 样本")

    # DataLoader
    nw = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=use_cuda, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=use_cuda)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=use_cuda)

    # 模型构建
    if use_dual_stream:
        print(f"\n使用双流架构: lightweight={args.lightweight}, fusion={args.fusion_type}")
        model = build_dual_stream_model(
            n_channels=base_n_ch,
            dropout=args.dropout,
            arch=args.arch,
            fusion_type=args.fusion_type,
            use_attention=(args.fusion_type == "attention"),
            lightweight=args.lightweight,
        )
    else:
        model = build_model(n_channels=n_ch, dropout=args.dropout, arch=args.arch)

    model = model.to(device)

    use_compile = use_cuda and not args.no_compile
    if use_compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile 已启用（mode={args.compile_mode}）")
        except Exception as e:
            print(f"torch.compile 失败: {e}")
            use_compile = False

    need_scaler = use_amp and not args.bf16
    scaler = torch.amp.GradScaler("cuda", enabled=need_scaler) if need_scaler else None

    criterion = nn.HuberLoss(delta=args.huber_delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: warmup_cosine_lr(e, args.warmup_epochs, args.epochs),
    )

    print(f"\n配置摘要:")
    print(f"  架构: {'双流' if use_dual_stream else '标准'} ({args.arch})")
    print(f"  多尺度增强: {args.scale_augment} (范围: {args.scale_range})")
    print(f"  Mixup: α={args.mixup_alpha}")
    print(f"  AMP: {use_amp} ({amp_dtype_name})")
    print(f"  FFT通道: {add_fft}")

    # 训练循环
    best_val_loss = float("inf")
    patience_cnt = 0
    train_losses, val_losses, val_maes = [], [], []

    print(f"\n开始训练（最多 {args.epochs} epoch，patience={args.patience}）")
    print(f'{"Epoch":>6}  {"Train Loss":>11}  {"Val Loss":>10}  {"Val MAE":>9}  {"时间":>6}')
    print("─" * 50)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=args.mixup_alpha,
            scaler=scaler, use_amp=use_amp, amp_dtype=amp_dtype,
            add_fft_channel=add_fft, grad_clip=args.grad_clip,
        )
        val_loss, val_mae = evaluate(
            model, val_loader, criterion, device,
            use_amp=use_amp, amp_dtype=amp_dtype, add_fft_channel=add_fft,
        )
        scheduler.step()

        dt = time.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_version": "v2",
                "arch": args.arch,
                "dual_stream": use_dual_stream,
                "n_channels": n_ch,
                "base_n_channels": base_n_ch,
                "add_fft_channel": add_fft,
                "dropout": args.dropout,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mae_deg": val_mae,
                "scale_augment": args.scale_augment,
                "scale_range": args.scale_range,
            }, model_path)
            flag = " ✓"
        else:
            patience_cnt += 1

        print(f"{epoch:>6}  {train_loss:>11.6f}  {val_loss:>10.6f}  {val_mae:>8.4f}°  {dt:>5.1f}s{flag}")

        if patience_cnt >= args.patience:
            print(f"\n早停触发（{args.patience} epoch）")
            break

    # 测试集评估
    print(f"\n加载最佳模型: {model_path}")
    ckpt = load_model_checkpoint(model_path, map_location=device)
    model.load_state_dict(state_dict_from_checkpoint(ckpt))

    test_loss, test_mae = evaluate(
        model, test_loader, criterion, device,
        use_amp=use_amp, amp_dtype=amp_dtype, add_fft_channel=add_fft,
    )
    print(f"测试集 MAE: {test_mae:.4f}°")

    # MC Dropout 评估
    if args.mc_samples > 0:
        print(f"\nMC Dropout 评估（{args.mc_samples} 采样）...")
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

        # 分层评估
        metrics = compute_stratified_metrics(all_means, all_labels, THETA_MIN, THETA_MAX)
        print(f"\n分层评估指标:")
        print(metrics.summary())

        # 校准评估
        calibration = compute_calibration_metrics(all_means, all_labels, all_stds)
        print(f"\n不确定性校准:")
        print(f"  1σ 覆盖率: {calibration['within_1sigma_pct']:.1f}% (理想 68.3%)")
        print(f"  2σ 覆盖率: {calibration['within_2sigma_pct']:.1f}% (理想 95.4%)")
        print(f"  误差-不确定性相关系数: {calibration['error_uncertainty_correlation']:.3f}")

        # 生成完整报告
        generate_evaluation_report(
            all_means, all_labels, out_dir,
            method_name="CNN_v2",
            uncertainties=all_stds,
        )

    print("\n" + "=" * 60)
    print(f"测试集 MAE: {test_mae:.4f}°")
    print(f"模型保存: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()