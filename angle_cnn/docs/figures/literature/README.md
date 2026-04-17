# 文献图片本体与版权说明

本目录用于存放可用于答辩对照的文献图片本体（或其本地快照），并记录来源与许可信息。

## 当前已落地文件

- `arxiv1911/weston2020_page_01.png` ... `arxiv1911/weston2020_page_08.png`  
  来源：Weston et al., *Atomic reconstruction in twisted bilayers of transition metal dichalcogenides*  
  原文页面：<https://arxiv.org/abs/1911.12664>  
  原始 PDF：`pmc_sources/arxiv_1911.12664.pdf`（仓库内已保存）

- `pmc10603799/nanolett2023_fig1_stm_fft.jpg`  
  来源：Nano Lett. 2023（PMC10603799）Figure 1（不同扭角 STM + FT）

- `pmc10603799/nanolett2023_fig3_spatial_modulation.jpg`  
  来源：Nano Lett. 2023（PMC10603799）Figure 3（moire 空间调制图）

- `pmc10603799/nanolett2023_fig2_spectra.jpg`  
  来源：Nano Lett. 2023（PMC10603799）Figure 2（谱学对比，辅助）

- `pmc10603799/nanolett2023_fig4_linecut.jpg`  
  来源：Nano Lett. 2023（PMC10603799）Figure 4（线切与角度趋势，辅助）

- `pmc7195481/liao2020_fig2_afm.jpg`  
  来源：Nat. Commun. 2020（PMC7195481）Figure 2（含 AFM 形貌图）

- `image_sources.json`  
  记录上述图片的下载链接、状态与文件路径（便于追溯）。

## 使用说明

- `arxiv1911/` 下图片是由原始 PDF 页面渲染得到（用于答辩展示时快速调用）。
- 你可在 PPT 中标注为“文献页面截图（来源见图注）”。

## 许可与合规

- 本目录优先收录开放获取或允许学术展示的内容。
- 对于需要平台验证/反爬的站点（如部分 Nature/PNAS 图页），自动抓取可能失败。  
  已在 `angle_cnn/docs/literature_image_benchmark.md` 保留可访问图页链接与出处说明，供你手动补充下载。
- 建议在论文/答辩中始终标注：作者、期刊、年份、图号与 URL。

## 后续补充建议（手动）

1. 打开 `angle_cnn/docs/literature_image_benchmark.md` 的图页链接。  
2. 手动下载图并放入本目录（建议命名：`author_year_figX.ext`）。  
3. 在本文件追加“来源 + 许可 + 使用场景”。
