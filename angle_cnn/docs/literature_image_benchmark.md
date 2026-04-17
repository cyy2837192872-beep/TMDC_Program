# 文献图像对照（用于答辩与实验设计）

本文件汇总与本项目 `MoS2 moire 仿真/识别` 最相关的前人图像证据。  
目标是回答两个问题：

1. **真实仪器是否能观测到与仿真相似的图样？**
2. **哪些图最适合作为你答辩时的“实证先例”？**

> 说明：本文件优先收录可直接访问的图页链接，并给出“与本项目输出的对应关系”。
>
> 本地图片本体（已下载/渲染）见：`angle_cnn/docs/figures/literature/`

---

## A. 与仿真最接近（优先展示）

### A1) STM moire 实空间纹理 + 频域峰（强对应 FFT 流程）

- **论文**: *Twist-Angle-Dependent Electronic Properties of Exfoliated Single Layer MoS2 on Au(111)*, Nano Letters (2023, open access)
- **图页（Figure 1）**:  
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC10603799/figure/fig1/>
- **图页（Figure 3）**:  
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC10603799/figure/fig3/>
- **本地图片本体（已下载）**:  
  `angle_cnn/docs/figures/literature/pmc10603799/nanolett2023_fig1_stm_fft.jpg`  
  `angle_cnn/docs/figures/literature/pmc10603799/nanolett2023_fig3_spatial_modulation.jpg`
- **全文**:  
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC10603799/>

**为什么与本项目相似**  
- 显示不同扭角下的 moire 周期变化（对应你的 `theta -> period` 物理关系）  
- 含 FT/频域峰信息（对应你的 FFT 基线流程）  
- 可直接支持“真实探针显微可得到与仿真同类周期纹理”

---

### A2) 小角度重构后的三角畴网络（强对应你的重构模型）

- **论文**: *Atomic reconstruction in twisted bilayers of transition metal dichalcogenides*, Nature Nanotechnology (2020)
- **arXiv 页面**:  
  <https://arxiv.org/abs/1911.12664>
- **PDF（含图）**:  
  <https://arxiv.org/pdf/1911.12664>
- **本地图片快照**（由 PDF 渲染）：  
  `angle_cnn/docs/figures/literature/arxiv1911/weston2020_page_01.png` 至 `weston2020_page_08.png`

**为什么与本项目相似**  
- 小角度下出现三角域/畴壁网络  
- 与你仿真中 `AB/BA` 畴对比、重构强度随角度变化的设定一致  
- 可作为“模型物理合理性”证据

---

### A3) 边缘角度区间（< 3°）下 moire 势与重构（补充）

- **论文**: *Moiré Potential, Lattice Relaxation, and Layer Polarization in Marginally Twisted MoS2 Bilayers*, Nano Letters (2023)
- **出版页**:  
  <https://doi.org/10.1021/acs.nanolett.2c03777>
- **项目内参考条目**:  
  `thesis/hust_bachelor_cse/Bibs/mybib.bib` 中 `tilak2023moire`

**为什么建议补充**  
- 直接讨论小角度扭转 MoS2 的 moire 势与晶格重构  
- 可用于支撑你“0.5°-1.5° 是优先实验窗口”的论证  
- 与你仿真中重构强度随角度衰减的物理设定一致

---

## B. AFM 路线直接先例（仪器可达性证据）

### B1) AFM 形貌图（MoS2 扭转样品可测）

- **论文**: *Precise control of the interlayer twist angle in large scale MoS2 homostructures*, Nature Communications (2020)
- **图页（Fig. 2，含 AFM）**:  
  <https://www.nature.com/articles/s41467-020-16056-4/figures/2>
- **PMC 全文**:  
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC7195481/>
- **本地图片本体（已下载，来自 PMC）**:  
  `angle_cnn/docs/figures/literature/pmc7195481/liao2020_fig2_afm.jpg`

**答辩可用表述**  
- “该工作在 MoS2 扭转样品上给出了 AFM 实测形貌，证明本项目的 AFM 成像路线在实验上可行。”

---

### B2) AFM 家族模式对 moire 的直接可视化

- **论文**: *Visualization of moiré superlattices*, Nature Nanotechnology (2020)
- **文章页**:  
  <https://www.nature.com/articles/s41565-020-0708-3>

- **论文**: *Torsional force microscopy of van der Waals moirés and atomic lattices*, PNAS (2024)
- **文章页**:  
  <https://www.pnas.org/doi/10.1073/pnas.2314083121>

**答辩可用表述**  
- “尽管模式不同于常规 Tapping Height，但这些 AFM 衍生模式已展示 moire 结构的可观测性，为本项目路线提供强外部支撑。”

---

### B3) 作为“AFM可用但不等同STM细节”的补充说法

- **论文**: *Precise control of the interlayer twist angle in large scale MoS2 homostructures*, Nat. Commun. (2020)
  - AFM 形貌主要用于样品质量与界面确认（非直接角度回归）
- **论文**: *Visualization of moiré superlattices*, Nat. Nanotechnol. (2020)
  - PFM 模式可视化 moire 网络（AFM 家族强证据）

**答辩关键词**  
- “AFM 可以稳定拿到周期/域结构层面的信息”  
- “STM 更偏原子/电子态细节，不是本项目第一阶段必要条件”  
- “本项目目标是转角提取，而非复现 STM 细节”

---

## C. 建议你在答辩 PPT 的排版方式

建议做一页“仿真 vs 文献”三栏对照：

1. **左栏（本项目仿真图）**：`height / phase` 示例 + 对应角度  
2. **中栏（文献图）**：A1 Figure 1（STM）或 A2 重构图  
3. **右栏（结论）**：  
   - 同类周期纹理存在  
   - 小角度出现域结构  
   - AFM/扫描探针路线已被前人验证

---

## D. 引用时建议措辞（可直接复制）

“本工作并非脱离物理实际的纯数值拟合。前人已在 MoS2 及相关扭转体系中通过 STM/AFM/PFM/TFM 观测到 moire 实空间纹理与重构域网络。本文在此可观测性基础上，进一步构建面向 Tapping 模式多通道数据的自动转角提取 pipeline，并将在后续实验中完成实测闭环验证。”

---

## E. 建议继续补充进仓库的图（待你手动下载）

> 受部分站点防护限制，以下图建议你手动下载后放入 `angle_cnn/docs/figures/literature/`。

1. **Nano Lett. 2023 (PMC10603799)**
   - Figure 1（moire 纹理 + FT）
   - Figure 3（空间调制图）
2. **Nat. Commun. 2020 (PMC7195481)**
   - Fig. 2（AFM 形貌 + 结构表征）
3. **Nat. Nanotechnol. 2020 (McGilly)**
   - 主图中 PFM 可视化 moire 网络图（用于 phase 通道论证）
4. **PNAS 2024 (Engelke)**
   - TFM 直接成像 moire / 原子纹理示例图

建议命名方式：`author_year_figX.ext`，例如 `liao2020_fig2.png`。
