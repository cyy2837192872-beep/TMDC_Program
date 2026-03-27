#!/usr/bin/env python3
"""
train_cnn.py — MoS₂ moiré 转角 CNN 回归训练
=============================================

任务：输入 128×128 MoS₂ moiré 图像，输出转角 θ（度）

模型：ResNet-18 改造版
  - 输入通道改为 1（灰度）
  - 最后全连接层改为输出 1 个值（回归）
  - 输出层接 Sigmoid，对应归一化后的 θ

训练策略
--------
- 损失函数：Huber Loss（比 MSE 对离群样本更鲁棒）
- 优化器：AdamW + CosineAnnealingLR
- 早停：val loss 连续 15 epoch 不下降则停止
- 最佳模型自动保存

运行方式
--------
    python train_cnn.py

输出
----
    outputs/best_model.pt     — 最佳验证集模型权重
    outputs/train_log.png     — 训练曲线
    outputs/train_log.csv     — 数值日志
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import csv
import matplotlib.font_manager as fm
from matplotlib import rcParams

myfont_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(myfont_path)
font_name = fm.FontProperties(fname=myfont_path).get_name()
rcParams['font.sans-serif'] = [font_name]
rcParams['axes.unicode_minus'] = False

# ── 路径 ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'data')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, 'moire_dataset.npz')
MODEL_PATH   = os.path.join(OUT_DIR,  'best_model.pt')
LOG_CSV      = os.path.join(OUT_DIR,  'train_log.csv')
LOG_PNG      = os.path.join(OUT_DIR,  'train_log.png')

# ── 超参数 ────────────────────────────────────────────────
BATCH_SIZE   = 64
NUM_EPOCHS   = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 15
NUM_WORKERS  = 4
THETA_MIN    = 0.5   # MoS₂ 转角范围下限（度）
THETA_MAX    = 5.0   # MoS₂ 转角范围上限（度）


# ── 设备选择 ──────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f'使用 GPU: {torch.cuda.get_device_name(0)}')
    else:
        dev = torch.device('cpu')
        print('使用 CPU（建议用 GPU 加速）')
    return dev


# ── 数据集 ────────────────────────────────────────────────

class MoireDataset(Dataset):
    """
    MoS₂ moiré 图像数据集

    标签归一化：θ_norm = (θ - THETA_MIN) / (THETA_MAX - THETA_MIN)
    """
    def __init__(self, images, labels, augment=False):
        self.images  = torch.from_numpy(images).unsqueeze(1)  # (N,1,H,W)
        self.labels  = torch.from_numpy(
            (labels - THETA_MIN) / (THETA_MAX - THETA_MIN)
        ).float()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].float()
        lbl = self.labels[idx]

        if self.augment:
            # 随机水平/垂直翻转（moiré 六角图案对称，翻转不改变角度）
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[1])
            # 随机90°旋转
            k = torch.randint(0, 4, (1,)).item()
            img = torch.rot90(img, k, dims=[1, 2])

        return img, lbl


# ── 模型 ──────────────────────────────────────────────────

def build_model():
    """
    ResNet-18 改造为灰度图回归模型

    改动：
    1. conv1：3通道 → 1通道（灰度输入）
    2. fc：1000类 → 1个值（θ 回归）
    3. 输出层后接 Sigmoid，输出 [0,1] 对应归一化后的 θ
    """
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    model.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

    return model


# ── 训练工具 ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, lbls)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(imgs)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """返回 loss 和 MAE（度）"""
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device).unsqueeze(1)

        preds = model(imgs)
        loss  = criterion(preds, lbls)
        total_loss += loss.item() * len(imgs)

        preds_deg = preds.squeeze(1) * (THETA_MAX - THETA_MIN) + THETA_MIN
        lbls_deg  = lbls.squeeze(1)  * (THETA_MAX - THETA_MIN) + THETA_MIN
        total_mae += (preds_deg - lbls_deg).abs().sum().item()

    n = len(loader.dataset)
    return total_loss / n, total_mae / n


def save_log_plot(train_losses, val_losses, val_maes):
    """保存训练曲线图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(r'MoS$_2$ moiré CNN 训练曲线', fontsize=13)

        ax1.plot(epochs, train_losses, 'b-', lw=1.5, label='train loss')
        ax1.plot(epochs, val_losses,   'r-', lw=1.5, label='val loss')
        ax1.set(xlabel='Epoch', ylabel='Huber Loss', title='Loss 曲线')
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(epochs, val_maes, 'g-', lw=1.5)
        ax2.axhline(0.1, color='red', ls='--', lw=1.2, label='目标 0.1°')
        ax2.set(xlabel='Epoch', ylabel='MAE (°)', title='验证集角度误差')
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(LOG_PNG, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'  [绘图跳过: {e}]')


# ── 主程序 ────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('MoS₂ moiré CNN 训练 — ResNet-18 角度回归')
    print('=' * 60)

    device = get_device()

    print(f'\n加载数据集: {DATASET_PATH}')
    data = np.load(DATASET_PATH)
    train_ds = MoireDataset(data['images_train'], data['labels_train'], augment=True)
    val_ds   = MoireDataset(data['images_val'],   data['labels_val'],   augment=False)
    test_ds  = MoireDataset(data['images_test'],  data['labels_test'],  augment=False)

    print(f'  train: {len(train_ds)} 样本')
    print(f'  val:   {len(val_ds)} 样本')
    print(f'  test:  {len(test_ds)} 样本')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    model = build_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n模型参数量: {total_params/1e6:.2f} M')

    criterion = nn.HuberLoss(delta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01
    )

    print(f'\n开始训练（最多 {NUM_EPOCHS} epoch，早停 patience={PATIENCE}）')
    print(f'{"Epoch":>6}  {"Train Loss":>11}  {"Val Loss":>10}  '
          f'{"Val MAE":>9}  {"LR":>9}  {"时间":>6}')
    print('─' * 62)

    best_val_loss = float('inf')
    patience_cnt  = 0
    train_losses, val_losses, val_maes = [], [], []

    with open(LOG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae_deg', 'lr'])

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss        = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        flag = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), MODEL_PATH)
            flag = ' ✓'
        else:
            patience_cnt += 1

        print(f'{epoch:>6}  {train_loss:>11.6f}  {val_loss:>10.6f}  '
              f'{val_mae:>8.4f}°  {lr:>9.2e}  {dt:>5.1f}s{flag}')

        if epoch % 10 == 0:
            save_log_plot(train_losses, val_losses, val_maes)

        with open(LOG_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_mae, lr])

        if patience_cnt >= PATIENCE:
            print(f'\n早停触发（{PATIENCE} epoch val loss 未改善）')
            break

    print(f'\n加载最佳模型: {MODEL_PATH}')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_loss, test_mae = evaluate(model, test_loader, criterion, device)
    print(f'测试集 MAE: {test_mae:.4f}°')
    print(f'测试集 Loss: {test_loss:.6f}')

    save_log_plot(train_losses, val_losses, val_maes)
    print(f'\n训练曲线: {LOG_PNG}')
    print(f'数值日志: {LOG_CSV}')
    print(f'模型权重: {MODEL_PATH}')

    print('\n' + '=' * 60)
    print(f'CNN 测试集 MAE:  {test_mae:.4f}°')
    print(f'FFT baseline:    ~0.05°（无畸变仿真）')
    print(f'FFT+畸变预期:    >1°')
    print('下一步：python eval_compare.py（CNN vs FFT 对比）')
    print('=' * 60)