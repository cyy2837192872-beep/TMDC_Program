# AFM 测试需求（精简版）

用于请老师快速判断：当前机器是否适合做该类样品扫描，以及大致机时和可行流程。

---

## 1) 样品与目标

- 样品：MoS2 转角同质结（小角度，约 0.5°-2°）
- 目标：获取可识别 moire 周期结构的 AFM 数据（以 Height 为核心）
- 用途：后续做转角识别算法验证（不涉及现场复杂分析）

---

## 2) 期望扫描条件（可协商）

| 项目 | 需求 |
|---|---|
| 扫描模式 | Tapping Mode |
| 探针 | TITAN 70 或同等级（tip radius 约 7 nm） |
| 扫描范围 | 200 nm-1 um（至少覆盖 3-5 个 moire 周期） |
| 分辨率 | 推荐 512x512（最低可 256x256） |
| 扫描速率 | 0.5-2 Hz（按机器稳定参数为准） |

---

## 3) 数据通道与导出

### 必需
- Height（核心）

### 最好同时有
- Phase
- Amplitude

### 导出格式
- 首选：`.spm`（保留完整元数据）
- 可选：`.tiff` / `.png` / `.csv`

---

## 4) 最少数据量建议

- 每个样品：至少 3 个位置
- 每个位置：建议 1-2 个扫描尺寸（例如 300 nm、500 nm）
- 目标：先拿到一批可用数据验证可行性，再决定是否扩量

---

## 5) 希望您帮忙确认

1. 该样品在现有机器上是否能稳定拍到可辨识的 moire 图案？
2. 以上参数里哪些需要按设备经验调整（尤其探针/速率/扫描尺寸）？
3. 是否支持导出 `.spm` 及批量导出图像/数值格式？
4. 按您经验，单个样品完成一轮扫描大约需要多少机时？
5. 近期是否可以安排先做一轮小规模测试？

---

## 6) 文献依据（供沟通时参考）

- Liao et al., *Nature Communications* (2020): Precise control of the interlayer twist angle in large scale MoS2 homostructures  
  含 AFM 形貌图（Fig. 2），可作为“MoS2 扭转样品可进行 AFM 表征”的先例  
  图链接：<https://www.nature.com/articles/s41467-020-16056-4/figures/2>  
  开放全文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7195481/>

- McGilly et al., *Nature Nanotechnology* (2020): Visualization of moiré superlattices  
  采用 PFM（AFM 家族模式）可视化 moiré 超晶格实空间结构  
  链接：<https://www.nature.com/articles/s41565-020-0708-3>

- Engelke et al., *PNAS* (2024): Torsional force microscopy of van der Waals moirés and atomic lattices  
  AFM 衍生模式（TFM）可实现 moiré/原子晶格成像  
  链接：<https://www.pnas.org/doi/10.1073/pnas.2314083121>

说明：上述文献用于支撑“扫描探针路线具备 moiré 可观测性”；本项目最终仍需在本机型与本样品条件下完成实测验证与误差闭环。