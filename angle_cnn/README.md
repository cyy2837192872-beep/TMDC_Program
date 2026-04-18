# angle_cnn — 运行说明（模块入口）

完整管线、文献索引与论文编译见仓库根目录 **[`../README.md`](../README.md)** 与 **[`../CHANGELOG.md`](../CHANGELOG.md)**。

本文档面向：**已克隆仓库**，希望在 **`angle_cnn/`** 目录下直接运行脚本的人或自动化代理。

---

## 前置条件

1. **Python ≥ 3.9**，已安装 PyTorch（见根目录 `requirements.txt` / `pyproject.toml`）。
2. **数据集**：默认路径为 **`../data/moire_dataset.npz`**（相对本目录）。若不存在，先在 `angle_cnn/` 下执行：

   ```bash
   python dataset_generator.py
   ```

3. **GPU**：训练与 `eval_compare` 推荐 CUDA；CPU 可跑但很慢。

---

## 最短工作流（复制即用）

在 **`tmdc-project/angle_cnn`** 下：

```bash
# 1) 训练（默认写入 ./outputs/）
python train_cnn.py --help

# 内存紧张（如 WSL）：减小 batch、单进程 DataLoader、半精度存图、可关 compile
python train_cnn.py --fp16-data --num-workers 0 --batch-size 256 --no-compile

# 可选 v5：EMA + 高角加权 loss（默认均为 0，与旧行为一致）
python train_cnn.py --fp16-data --num-workers 0 --batch-size 256 --no-compile \
  --ema-decay 0.999 --angle-loss-weight 0.25

# 2) 仅重跑测试 + MC（需已有 outputs/best_model.pt）
python train_cnn.py --eval-only --mc-samples 30

# 3) FFT vs CNN 公平对比（需 best_model.pt）
python eval_compare.py --mc-samples 30

# 4) 论文用合并表（需 train_test_summary.csv 与 compare_summary.csv）
python export_thesis_table.py
```

输出默认在 **`./outputs/`**（多数大文件被根目录 `.gitignore` 忽略，需本地生成）。

---

## 常见故障

| 现象 | 处理 |
|------|------|
| 进程被系统 `Killed`、无 Python 栈 | 多为 **RAM 不足**；用 `--fp16-data`、`--batch-size 128/256`、`--num-workers 0`、减小 WSL 外 `.wslconfig` 或增大 swap |
| `torch.compile` 报错 / 缺编译器 | 加 **`--no-compile`**，或安装 `build-essential`（gcc） |
| `FileNotFoundError: moire_dataset.npz` | 确认 **`../data/`** 下已有 npz，或 `dataset_generator.py` 生成路径与 `--data-dir` 一致 |
| `eval_compare` 与论文数字不一致 | 确认 **数据集** 为当前固定 FOV 代码生成；**权重** 与 **脚本版本** 与 `CHANGELOG.md` 一致 |

---

## 测试

```bash
cd angle_cnn
pip install -e "..[dev]"   # 若需 pytest
pytest tests -q
# 无 pytest 时：
python tests/test_train_utils.py
```
