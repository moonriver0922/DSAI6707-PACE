# 项目方向（交付收尾）

**生效**：2026-10-26  
**主交付叙事**：`reports/PACE_Course_Report.md` + `reports/PACE_Presentation_6slides.md`  
**规范（实现）**：`PACE_METHOD_SPEC.md` / `PACE_Research_Implementation_Plan.md`

## 一句话故事

低预算进化 Prompt 搜索容易因噪声过早淘汰好候选；**PACE** 用配对置信证据做生存选择；离线错误淘汰明显下降；在线 15-seed 准确率有正向趋势但未锁定显著优势。

## 冻结（不再改）

- Gate 1 / Gate 2 协议与超参  
- 15-seed 主结果与消融  
- 不新开 GPU、不上 SPaCE、不扩模型/数据  

## 历史材料（不进主叙事）

- DICE-PO 负结果：`reports/DICE_PO_Technical_Report.md`、`outputs/gate1/`  
- 机制审计 / SPaCE 讨论：`outputs/pace/gate2_cap_align_15seed/analysis/`  

## 剩余交付动作

1. 将 `PACE_Course_Report.md` 导出为课程 PDF（可本地 pandoc/Word）  
2. 按 6-slide 提纲做 Presentation  
3. 打包代码 + `outputs/pace/` 关键结果一并提交  
