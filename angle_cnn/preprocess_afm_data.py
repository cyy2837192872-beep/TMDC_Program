#!/usr/bin/env python3
"""
preprocess_afm_data.py — AFM 实验数据预处理脚本
=================================================

将 AFM 实验数据转换为项目兼容格式（.npz），供 CNN 模型推理使用。

支持的数据格式
--------------
1. .spm 文件（Asylum Research 原始格式）— 推荐
2. .tiff / .png 图像文件
3. .csv / .txt 数值文件
4. .ibw 文件（Igor Binary Wave）

输出格式
--------
- images: (N, C, 128, 128) float32，归一化到 [0, 1]
- metadata: 扫描参数和元数据

运行方式
--------
    # 处理单个文件
    python preprocess_afm_data.py --input data/afm_raw/sample1.spm --output data/afm_processed/

    # 批量处理目录
    python preprocess_afm_data.py --input-dir data/afm_raw/ --output data/afm_processed/

    # 生成可视化预览
    python preprocess_afm_data.py --input data/afm_raw/ --output data/afm_processed/ --preview
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


@dataclass
class AFMMetadata:
    """AFM 图像元数据。"""
    filename: str
    scan_size_nm: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    scan_rate_hz: float = 0.0
    probe_type: str = ""
    tip_radius_nm: float = 0.0
    channels: List[str] = None
    sample_id: str = ""
    position_id: str = ""
    notes: str = ""
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = []


def load_spm_file(filepath: str) -> Tuple[Dict[str, np.ndarray], AFMMetadata]:
    """加载 Asylum Research .spm 文件。
    
    需要安装 pyAsylum 或 spm_reader 库。
    如果不可用，尝试使用 IGOR pro 或导出为其他格式。
    """
    metadata = AFMMetadata(filename=os.path.basename(filepath))
    channels = {}
    
    try:
        # 尝试使用 pyAsylum
        import pyAsylum as pa
        
        spm = pa.SPMFile(filepath)
        
        # 提取通道数据
        if hasattr(spm, 'height'):
            channels['height'] = spm.height.data.astype(np.float32)
        if hasattr(spm, 'phase'):
            channels['phase'] = spm.phase.data.astype(np.float32)
        if hasattr(spm, 'amplitude'):
            channels['amplitude'] = spm.amplitude.data.astype(np.float32)
        
        # 提取元数据
        if hasattr(spm, 'scan_size'):
            metadata.scan_size_nm = float(spm.scan_size) * 1e9  # m -> nm
        if hasattr(spm, 'scan_rate'):
            metadata.scan_rate_hz = float(spm.scan_rate)
        
    except ImportError:
        # 尝试使用 spm_reader
        try:
            from spm_reader import read_spm
            
            data = read_spm(filepath)
            
            # 假设返回格式
            if isinstance(data, dict):
                if 'height' in data:
                    channels['height'] = np.array(data['height'], dtype=np.float32)
                if 'phase' in data:
                    channels['phase'] = np.array(data['phase'], dtype=np.float32)
                if 'amplitude' in data:
                    channels['amplitude'] = np.array(data['amplitude'], dtype=np.float32)
                if 'metadata' in data:
                    meta = data['metadata']
                    metadata.scan_size_nm = float(meta.get('scan_size_nm', 0))
                    metadata.scan_rate_hz = float(meta.get('scan_rate_hz', 0))
                    
        except ImportError:
            logging.warning(
                f"无法解析 .spm 文件: {filepath}\n"
                "请安装 pyAsylum 或 spm_reader，或先将文件导出为 .tiff/.csv 格式。\n"
                "安装命令: pip install spm-reader"
            )
            return None, metadata
    
    metadata.channels = list(channels.keys())
    if channels:
        metadata.resolution = list(channels.values())[0].shape
    
    return channels, metadata


def load_ibw_file(filepath: str) -> Tuple[Dict[str, np.ndarray], AFMMetadata]:
    """加载 Igor Binary Wave (.ibw) 文件。"""
    metadata = AFMMetadata(filename=os.path.basename(filepath))
    channels = {}
    
    try:
        import igor2.binarywave as ibw
        
        data = ibw.load(filepath)
        
        if 'wave' in data:
            wave = data['wave']
            if 'wData' in wave:
                raw = wave['wData']
                
                # IBW 可能包含多通道数据
                if raw.ndim == 3:
                    # 假设形状为 (channels, height, width)
                    channel_names = ['height', 'phase', 'amplitude']
                    for i in range(min(raw.shape[0], len(channel_names))):
                        channels[channel_names[i]] = raw[i].astype(np.float32)
                elif raw.ndim == 2:
                    channels['height'] = raw.astype(np.float32)
            
            # 提取元数据
            if 'note' in wave:
                note = wave['note']
                if isinstance(note, dict):
                    metadata.scan_size_nm = float(note.get('ScanSize', 0))
                    metadata.scan_rate_hz = float(note.get('ScanRate', 0))
                    metadata.probe_type = str(note.get('ProbeType', ''))
        
    except ImportError:
        logging.warning(
            f"无法解析 .ibw 文件: {filepath}\n"
            "请安装 igor2: pip install igor2"
        )
        return None, metadata
    
    metadata.channels = list(channels.keys())
    if channels:
        metadata.resolution = list(channels.values())[0].shape
    
    return channels, metadata


def load_image_file(filepath: str) -> Tuple[Dict[str, np.ndarray], AFMMetadata]:
    """加载图像文件（.tiff, .png, .bmp 等）。"""
    metadata = AFMMetadata(filename=os.path.basename(filepath))
    channels = {}
    
    try:
        from PIL import Image
        
        img = Image.open(filepath)
        arr = np.array(img, dtype=np.float32)
        
        # 处理不同维度
        if arr.ndim == 2:
            # 灰度图像
            channels['height'] = arr
        elif arr.ndim == 3:
            if arr.shape[2] == 3:
                # RGB 图像，假设三个通道分别对应 height/phase/amplitude
                channels['height'] = arr[:, :, 0]
                channels['phase'] = arr[:, :, 1]
                channels['amplitude'] = arr[:, :, 2]
            elif arr.shape[2] == 4:
                # RGBA，忽略 alpha 通道
                channels['height'] = arr[:, :, 0]
                channels['phase'] = arr[:, :, 1]
                channels['amplitude'] = arr[:, :, 2]
        
    except ImportError:
        # 尝试使用 imageio 或 cv2
        try:
            import imageio
            arr = imageio.imread(filepath).astype(np.float32)
            
            if arr.ndim == 2:
                channels['height'] = arr
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                channels['height'] = arr[:, :, 0]
                channels['phase'] = arr[:, :, 1]
                channels['amplitude'] = arr[:, :, 2]
                
        except ImportError:
            logging.warning(f"无法加载图像文件: {filepath}")
            return None, metadata
    
    metadata.channels = list(channels.keys())
    if channels:
        metadata.resolution = list(channels.values())[0].shape
    
    return channels, metadata


def load_csv_file(filepath: str) -> Tuple[Dict[str, np.ndarray], AFMMetadata]:
    """加载 CSV/TXT 数值文件。"""
    metadata = AFMMetadata(filename=os.path.basename(filepath))
    channels = {}
    
    try:
        # 尝试检测格式
        with open(filepath, 'r') as f:
            first_line = f.readline()
            f.seek(0)
            
            # 检查是否有表头
            has_header = any(c.isalpha() for c in first_line if not c.isspace())
            
        # 加载数据
        if has_header:
            data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        else:
            data = np.loadtxt(filepath, delimiter=',')
        
        if data.ndim == 2:
            if data.shape[1] == 1:
                # 单列数据，尝试 reshape 为正方形
                n = int(np.sqrt(data.shape[0]))
                if n * n == data.shape[0]:
                    channels['height'] = data.reshape(n, n).astype(np.float32)
            elif data.shape[1] >= 3:
                # 可能是 (x, y, z) 格式
                # 这里简化处理，假设已经是 2D 格式
                channels['height'] = data.astype(np.float32)
        
    except Exception as e:
        logging.warning(f"无法加载 CSV 文件: {filepath}, 错误: {e}")
        return None, metadata
    
    metadata.channels = list(channels.keys())
    if channels:
        metadata.resolution = list(channels.values())[0].shape
    
    return channels, metadata


def load_afm_data(filepath: str) -> Tuple[Optional[Dict[str, np.ndarray]], AFMMetadata]:
    """自动检测文件格式并加载 AFM 数据。"""
    ext = Path(filepath).suffix.lower()
    
    if ext == '.spm':
        return load_spm_file(filepath)
    elif ext == '.ibw':
        return load_ibw_file(filepath)
    elif ext in ['.tiff', '.tif', '.png', '.bmp', '.jpg', '.jpeg']:
        return load_image_file(filepath)
    elif ext in ['.csv', '.txt', '.dat']:
        return load_csv_file(filepath)
    else:
        logging.warning(f"不支持的文件格式: {ext}")
        return None, AFMMetadata(filename=os.path.basename(filepath))


def preprocess_image(
    img: np.ndarray,
    target_size: int = 128,
    normalize: bool = True,
) -> np.ndarray:
    """预处理单通道图像。
    
    Parameters
    ----------
    img : np.ndarray
        输入图像，形状 (H, W)
    target_size : int
        目标尺寸
    normalize : bool
        是否归一化到 [0, 1]
    
    Returns
    -------
    np.ndarray
        预处理后的图像，形状 (target_size, target_size)
    """
    import torch
    import torch.nn.functional as F
    
    # 转换为 tensor
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    
    # 归一化
    if normalize:
        t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    
    # 调整尺寸
    if img.shape[0] != target_size or img.shape[1] != target_size:
        t = F.interpolate(t, size=(target_size, target_size), mode='bilinear', align_corners=False)
    
    return t.squeeze().numpy().astype(np.float32)


def preprocess_afm_dataset(
    channels: Dict[str, np.ndarray],
    metadata: AFMMetadata,
    target_size: int = 128,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """预处理 AFM 数据集。
    
    Parameters
    ----------
    channels : dict
        通道数据字典 {'height': arr, 'phase': arr, 'amplitude': arr}
    metadata : AFMMetadata
        元数据
    target_size : int
        目标图像尺寸
    
    Returns
    -------
    image : np.ndarray
        预处理后的图像，形状 (C, target_size, target_size)
    meta_dict : dict
        元数据字典
    """
    # 定义通道顺序（与训练数据一致）
    channel_order = ['height', 'phase', 'amplitude']
    
    processed = []
    for ch_name in channel_order:
        if ch_name in channels:
            img = preprocess_image(channels[ch_name], target_size)
            processed.append(img)
    
    if not processed:
        raise ValueError("没有有效的通道数据")
    
    # 如果只有一个通道，复制到三个通道（兼容多通道模型）
    while len(processed) < 3:
        processed.append(processed[0].copy())
    
    # 堆叠为多通道图像
    image = np.stack(processed[:3], axis=0)  # (3, target_size, target_size)
    
    # 转换元数据为字典
    meta_dict = asdict(metadata)
    
    return image, meta_dict


def generate_preview(
    channels: Dict[str, np.ndarray],
    metadata: AFMMetadata,
    save_path: str,
):
    """生成预览图。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        n_channels = len(channels)
        fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))
        
        if n_channels == 1:
            axes = [axes]
        
        for ax, (name, data) in zip(axes, channels.items()):
            im = ax.imshow(data, cmap='afmhot')
            ax.set_title(f"{name}\n{data.shape[0]}×{data.shape[1]}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        fig.suptitle(f"{metadata.filename}\nScan Size: {metadata.scan_size_nm:.1f} nm", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"预览图已保存: {save_path}")
        
    except ImportError:
        print("警告：matplotlib 不可用，跳过预览图生成")


def process_single_file(
    filepath: str,
    output_dir: str,
    target_size: int = 128,
    preview: bool = False,
) -> Optional[str]:
    """处理单个 AFM 文件。
    
    Returns
    -------
    str or None
        输出文件路径，失败返回 None
    """
    print(f"\n处理文件: {filepath}")
    
    # 加载数据
    channels, metadata = load_afm_data(filepath)
    
    if channels is None or len(channels) == 0:
        print(f"  警告：无法加载数据，跳过")
        return None
    
    print(f"  加载成功：{len(channels)} 个通道，分辨率 {metadata.resolution}")
    print(f"  通道：{list(channels.keys())}")
    
    # 生成预览
    if preview:
        preview_path = os.path.join(
            output_dir, 
            'preview', 
            Path(filepath).stem + '_preview.png'
        )
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)
        generate_preview(channels, metadata, preview_path)
    
    # 预处理
    try:
        image, meta_dict = preprocess_afm_dataset(channels, metadata, target_size)
        print(f"  预处理完成：形状 {image.shape}")
    except Exception as e:
        print(f"  预处理失败：{e}")
        return None
    
    # 保存
    output_path = os.path.join(output_dir, Path(filepath).stem + '.npz')
    np.savez_compressed(
        output_path,
        image=image,
        metadata=json.dumps(meta_dict),
    )
    
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  已保存: {output_path} ({size_kb:.1f} KB)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="AFM 实验数据预处理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, help="输入文件路径")
    parser.add_argument("--input-dir", type=str, help="输入目录（批量处理）")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--target-size", type=int, default=128, help="目标图像尺寸")
    parser.add_argument("--preview", action="store_true", help="生成预览图")
    parser.add_argument("--pattern", type=str, default="*", help="文件匹配模式")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    os.makedirs(args.output, exist_ok=True)
    
    # 收集输入文件
    input_files = []
    
    if args.input:
        input_files.append(args.input)
    
    if args.input_dir:
        for ext in ['*.spm', '*.ibw', '*.tiff', '*.tif', '*.png', '*.csv', '*.txt']:
            pattern = os.path.join(args.input_dir, '**', ext)
            import glob
            input_files.extend(glob.glob(pattern, recursive=True))
    
    if not input_files:
        print("错误：没有找到输入文件")
        return
    
    print(f"找到 {len(input_files)} 个文件")
    
    # 处理文件
    output_files = []
    for filepath in input_files:
        result = process_single_file(
            filepath, 
            args.output, 
            args.target_size,
            args.preview
        )
        if result:
            output_files.append(result)
    
    # 生成索引文件
    if output_files:
        index_path = os.path.join(args.output, 'index.json')
        index = {
            'files': [os.path.basename(f) for f in output_files],
            'n_samples': len(output_files),
            'target_size': args.target_size,
            'channels': ['height', 'phase', 'amplitude'],
        }
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"\n处理完成：{len(output_files)}/{len(input_files)} 个文件成功")
        print(f"索引文件: {index_path}")
        print(f"\n下一步：使用 inference_real_data.py 进行模型推理")


if __name__ == "__main__":
    main()