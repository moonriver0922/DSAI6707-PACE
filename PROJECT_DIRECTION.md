# 项目方向（对齐 PACE 计划）

**生效**：2026-10-23  
**规范文件**：`PACE_Research_Implementation_Plan.md`（v1.0）  
**旧规范**：DICE-PO Final Plan v3.0 → 结果保留为**历史负结果**，不再作为主攻方向。

## 新主线：PACE

**Working title:** *PACE: Paired-Assessment with Confidence-Preserving Evolution*

**核心机制：** Evidence-Gated Survivor Selection — 不确定性决定**是否有足够证据淘汰候选**，而不是给 pair 打 DICE 式 decision-impact 分。

**原则：**

1. 先用已有 `oracle_matrices.npz` 做 Gate 1a/1b（**0 新 LLM**），再决定是否上 Gate 2 在线进化。  
2. DICE 的负结果与事后公平消融**如实保留**，不得改贴成 PACE 胜利。  
3. 对照必须匹配：同一候选池、同一 Racing 评估、同一预算/种子/容量；**唯一处理**是是否证据门控淘汰。  
4. Gate 1 负结果同样可交付；禁止阈值扫描与偷看 test/oracle。

## 停止做什么

- 继续改 DICE 权重找胜利  
- 不公平的 NSGA vs Random 再解读为选择机制结论  
- 未过 Gate 1 就跑大规模 Gate 2

## 交付顺序

Gate 0 单测 → Gate 1a 错误淘汰是否存在 → Gate 1b PACE vs 匹配点估计基线 → GO/NO-GO →（仅 GO）Gate 2。
