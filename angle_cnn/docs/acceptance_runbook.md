# 修复后训练与真实数据验收

六项修复的自动化验证位于 `tests/test_acceptance_fixes.py` 与 `tests/test_resume_training.py`。

## 完整训练

在项目根目录运行，建议使用原 CUDA 训练机器。默认完整实验为 50000 个样本、3 通道、n_sim=512、seed=42、uniform 采样。训练默认计划 200 epochs，patience=30；达到早停也属于计划正常结束。先确认 GPU、内存与磁盘可用，记录 Python、PyTorch、CUDA 版本。

```bash
python -m angle_cnn.dataset_generator --samples 50000 --channels 3 --n-sim 512 --theta-sampling uniform --seed 42 --workers 4
python -m angle_cnn.train_cnn --no-compile --batch-size 256 --epochs 200 --patience 30 --num-workers 0 --mc-samples 30 --seed 42
python -m angle_cnn.eval_compare --eval-mode paired --paired-seed 12345 --mc-samples 30 --save-predictions
python -m angle_cnn.graded_eval --mc-samples 30
python -m angle_cnn.check_regression
```

`check_regression` 需要独立保存的正式基线，不应把待验收的新模型保存成自己的精度基线。如果基线对应的测试集不是本次固定数据版本，应先在本次测试集上复核旧模型，并明确记录数据 SHA-256；否则不能声称新旧指标可直接比较。

训练每轮原子写入 `angle_cnn/outputs/last_training.pt`，记录优化器、调度器、GradScaler、EMA/SWA、随机状态、DataLoader 生成器、历史指标、最佳模型以及数据 SHA-256。`best_model.pt` 用于推理，不能用于完整续训。

```bash
python -m angle_cnn.train_cnn --resume angle_cnn/outputs/last_training.pt
# 从旧模型重新训练／微调：新训练的结构和通道参数必须与权重匹配
python -m angle_cnn.train_cnn --init-weights angle_cnn/outputs/best_model.pt --no-compile
```

`--resume` 恢复原来的训练参数和总 epochs 计划；命令行可调整 data-dir、output-dir 和 stop-after-epoch，但数据内容必须匹配 SHA-256。若改变训练计划，用 init-weights 开始新实验。已经完成或早停的计划不能继续 resume。

`--stop-after-epoch N` 在指定轮次保存状态后退出，便于计划性中断；异常中断后从最近完整轮次继续。为可恢复 worker 随机状态，训练不保留跨轮次的 persistent workers。CPU 上已测试连续训练与跨进程恢复完全一致（包括多 worker、EMA/SWA、FFT 通道）；跨硬件与软件版本不保证逐位一致，CUDA AMP 的数值一致性仍需在实际训练机器复核。

## 真实数据验收

需要真实原始 IBW/TIFF 或受支持格式、可信扫描视野（nm）、对应参考角度、参考角度测量方法／不确定度，以及样本与扫描区域 ID。没有真实标注时，只能验收推理执行，不能验收角度精度。

建议标注表包含：`filename,sample_id,position_id,fov_nm,reference_angle_deg,reference_uncertainty_deg,split`。

1. 先选少量有明确通道名和扫描视野的 IBW，核对 height/phase/amplitude 图像方向、维度与 nm 单位。当前读取器支持方形 H×W×C 和 C×H×W；歧义维度会报错。Cypher note 的 ScanSize 按米转换为 nm。
2. 同一扫描分别走预处理 NPZ 和原始文件推理，核对物理裁剪视野、原图 FFT 输入和预测结果。
3. 用验证集选择融合 τ；按样本／区域划分独立测试集，避免同一扫描的裁剪图跨集合。
4. 固定模型、数据、seed 与参数，记录 CNN/FFT/融合 MAE、MedAE、P95、失败率、有效样本数及低／高角度分层指标。失败样本计入总样本数。
5. 保存模型和输入数据校验和、逐样本预测、运行命令与环境。推理的任一样本失败返回非零，并在 CSV 保留 ERROR 行。

```bash
python -m angle_cnn.preprocess_afm_data --input scan.ibw --output afm_processed
python -m angle_cnn.inference_real_data --input-dir afm_processed --model angle_cnn/outputs/best_model.pt --output real_results --seed 42 --mc-samples 30
```

本机的合成 TIFF 和模拟 IBW 解析测试不是实际仪器文件验收，也不替代完整训练与真实标注集验收。
