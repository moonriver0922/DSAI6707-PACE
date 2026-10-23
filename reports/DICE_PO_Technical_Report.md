# DICE-PO 技术报告

**Decision-Critical Paired Evaluation for Budgeted Evolutionary Multi-objective Prompt Optimization**

| 项目 | 内容 |
|------|------|
| 课程 | DSAI6707 Advanced Computational Intelligence |
| 版本 | 实验报告 v1.3（含事后公平消融 post-hoc fair ablation） |
| 日期 | 2026-10-23 |
| 硬件 | NVIDIA A100-SXM4-80GB（单卡推理） |
| 性质 | 预注册实验 + 真实结果；**非**已证实有效的 DICE 算法主张 |

---

## 摘要

在有限样本级评估预算下，多目标 Prompt 优化需要同时权衡任务准确率与部署 token 成本。本文实现并评测 **DICE-PO**：用配对翻转风险 \(r_{pq}\)、决策影响 \(I_{pq}\) 与边际评估成本构造查询优先级。

**Gate 1（DICE 主假设）**：AG News + Qwen2.5-7B-Instruct，16 冻结 Prompt，一次 5,760 推理 + 20 seed 离线回放。主预算 640 上 Test Preference Regret：Racing **0.0058** ≈ Uniform **0.0059** ≪ Full DICE **0.0141**。**H1/H2 不支持 → 停止 DICE 扩容（NO-GO）**。

**机制诊断（0 新推理）**：Racing 含早停+整列回退，DICE 无回退且预算更集中；Impact 改变了购买的 pair 集合却未降 regret。

**课程 EC 补课（不含 DICE）**：NSGA-II+Uniform / +Racing / Random Search，种群 8、4 代、3 seeds、每 run 1,280 calls。Winner test acc 均值：Random **0.825**、NSGA-II+Racing **0.792**、NSGA-II+Uniform **0.772**。证明进化搜索管线可运行；**3 seed 不足以声称 NSGA 优于随机**，且**不覆盖** DICE 负结果。

---

## 1. 引言

### 1.1 问题

冻结一个 instruction-tuned LLM，对分类任务进化/搜索候选 Prompt。每次 \((p,x)\) 调用得到对错与 token。目标：

\[
\max_p a(p),\quad \min_p c(p)
\]

其中 \(a(p)=\mathbb{E}[\mathbf{1}\{M(p,x)=y\}]\)，\(c(p)\) 为加权输入/输出 token 期望。搜索阶段另有硬预算：**search evaluation calls**（缓存命中计 0）。

### 1.2 主张与边界

**主张（待证伪）**：最值得购买的评估，不一定来自置信区间最宽的候选，而是那些可能翻转 Pareto 保留/淘汰或最终偏好选择的候选对。

**贡献边界**：多目标 Prompt 优化、Racing、配对估计已有先行工作（MO-CAPO、BudgetAPO、TRIPLE 等）。DICE 仅检验「风险 × 决策影响 / 边际成本」选择策略是否在固定预算下降低偏好选择后悔。

### 1.3 三项假设

| ID | 内容 | Gate 1 结论 |
|----|------|-------------|
| H1 | 同预算下 DICE 的 Test Preference Regret 低于 Paired Racing（主看 640） | **不支持** |
| H2 | 乘以 decision impact 优于仅用 pairwise risk | **不支持**（两者接近且均差于 Racing） |
| H3 | 收益主要出现在准确率接近、成本有权衡时 | 未单独析因；场景存在但仍无 DICE 优势 |

---

## 2. 相关工作（主题式，非逐篇堆砌）

正式课程报告需 ≥20 篇 2024–2026 文献的主题综述。本稿先界定近邻：

1. **预算感知 Prompt 优化**：BudgetAPO、CAPO 等强调评估成本与早停。
2. **多目标 Prompt / Pareto 搜索**：MO-CAPO 等将准确率与成本联合优化。
3. **Racing / 配对比较**：TRIPLE 等用共享样本降低方差、加速淘汰。
4. **主动/不确定性采样**：在 LLM 评测中常见，但易引入非均匀样本的总体估计偏差。

**Gap**：在「配对风险」之上显式乘以「偏好/Pareto 决策后果」，并在固定 call 预算下用 **test preference regret** 做主指标。本实验即对该 gap 的可证伪检验。

---

## 3. 方法（MVP 实现规格）

### 3.1 状态与两类样本角色

维护 \(z[p,i]\in\{0,1,\varnothing\}\)、真实 token、稳定 `prompt_id` / `sample_id`。  
策略**不得**读取 test oracle。

- **共同均匀 anchor**：每候选初始共享 8 个随机 search 样本（16×8=128 calls）。
- **后续查询**：Uniform 按列补全；配对策略为最高分 pair 追加 8 个**共同随机**新样本（至多 16 calls；缓存不计）。

v1 **不做**「最有鉴别力题目」主动选题，以避免 selection bias 与主机制耦合。

### 3.2 DICE 分数

对 contender（非支配 + 邻域，最多 6 个）按成本排序，取最多 8 个邻近对。对共同观测集做 200 次 paired bootstrap，得翻转率 \(r_{pq}\)（启发式风险，**非**校准后验）。决策影响 \(I_{pq}\)：在 \(\lambda\in\{0.25,0.5,0.75\}\) 上，假设 \(p,q\) 效用顺序反转后，赢家变化对应的归一化效用差均值。

\[
s(p,q)=\frac{r_{pq}\cdot I_{pq}}{\max(1,\Delta\mathrm{calls}_{pq})}
\]

**Risk-Only**：\(I_{pq}\equiv 1\)。**Paired Racing**：同类配对分配 + 低风险早停阈值。**Uniform**：均匀补样本。

### 3.3 效用与主指标

\[
U_\lambda(p)=\lambda a(p)-(1-\lambda)\tilde{c}(p),\quad \tilde{c}\in[0,1]
\]

成本归一化界限预先固定为 \([0,256]\)（search pool 物理代理，不读 test）。

**Preference Selection Regret（test）**：用 development 估计按 \(\lambda\) 选 \(p^*_{\mathrm{dev}}\)，与同一冻结候选池上的 test-oracle 赢家比较：

\[
R_\lambda=U^\mathrm{test}_\lambda(p^*_{\mathrm{oracle}})-U^\mathrm{test}_\lambda(p^*_{\mathrm{dev}})
\]

报告 \(\lambda\in\{0.25,0.5,0.75\}\) 及宏平均。Test 密封，仅最终离线计分。

### 3.4 软件架构

```
DICE_PO/
  configs/gate1.yaml, gate2.yaml
  data/prompt_pool.json, split_manifest.json
  src/{inference,dataset,cache,state,pareto,metrics,replay,evolution}
  src/policies/{uniform,racing,risk_only,dice}
  scripts/{build_splits,precompute_matrix,run_replay,report,smoke_gpu}
  tests/   # Gate 0
```

接口契约：`BudgetPolicy.select_queries(state, remaining_calls) -> list[(prompt_id, sample_id)]`。

---

## 4. 实验协议

### 4.1 Gate 0（工程正确性）

合成 6×20 矩阵单测：oracle 泄漏不变性、缓存不计预算、剩余 1 call 禁止完整新 pair、配对 sample ID、效用/HV 符号、确定性 seed。  
**结果**：15/15 通过。

### 4.2 Gate 1（主实验）

| 项 | 设定 |
|----|------|
| 数据 | AG News（`fancyzhx/ag_news`），search 160 / test 200，分层抽样，seed=6707 |
| 模型 | Qwen2.5-7B-Instruct，BF16，greedy，`max_new_tokens=8` |
| 候选 | 16 个冻结 Prompt（短指令 / 角色 / 格式 / 定义） |
| 预计算 | \(16\times(160+200)=5760\) 次，**仅一次**；之后纯离线 replay |
| 策略 | Uniform, Paired Racing, Risk-Only, Full DICE |
| 预算点 | 320 / **640（主）** / 1280 |
| Seeds | 20 个 replay seed（不新增 LLM 调用） |

GPU smoke：单条 Sports 样本正确分类；峰值显存约 15.3 GB。

### 4.3 Gate 2

规范要求证明可插入 NSGA-II 在线进化。因 Gate 1 **NO-GO**，**未执行**真实在线搜索（避免在无效机制上消耗约 1.3 万次额外调用）。代码侧保留 `evolution.py` 与 `run_online.py --go` 护栏。

---

## 5. 结果

### 5.1 冻结池上的经验景观（search / test）

全池 search 平均准确率 0.677，test 0.660。代表候选：

| ID | 类型要点 | search acc | test acc | search mean tokens |
|----|----------|------------|----------|---------------------|
| p15 | 角色+规范标签 | 0.856 | 0.805 | 117.7 |
| p05 | 角色 | 0.831 | 0.790 | 117.6 |
| p02 | 编辑角色 | 0.825 | 0.780 | 115.0 |
| p03 | 严格单标签格式 | 0.819 | 0.790 | 110.7 |
| p00 | 短直接 | 0.800 | 0.795 | 108.7 |
| p06 | JSON 格式 | **0.000** | **0.000** | 114.4 |

`p06` 在冻结解析器下始终无法解析为合法标签，准确率为 0；按协议**不事后替换** Prompt。高精度候选之间准确率接近、token 成本有分化，满足 H3 所述「接近准确率 + 成本权衡」的基本场景。

### 5.2 主表：预算 640 的宏平均 Test Preference Regret

| Policy | Mean | SD | 相对 Racing |
|--------|------|-----|-------------|
| **racing** | **0.00584** | 0.00540 | — |
| uniform | 0.00595 | 0.00313 | ≈持平 |
| risk_only | 0.01380 | 0.01944 | 更差 |
| **dice** | **0.01407** | 0.01549 | 更差（Δ≈+0.008） |

逐 seed：`mean(dice − racing) > 0`（DICE 后悔更高）。

### 5.3 预算曲线（宏平均）

| Budget | uniform | racing | risk_only | dice |
|--------|---------|--------|-----------|------|
| 320 | 0.01087 | 0.01009 | 0.01385 | 0.01398 |
| 640 | 0.00595 | **0.00584** | 0.01380 | 0.01407 |
| 1280 | 0.00712 | 0.00627 | 0.01373 | 0.01490 |

DICE 在三个注册预算点均未优于 Racing；1280 时 DICE 略升而非下降。Uniform 与 Racing 在中高预算接近。

### 5.4 分 \(\lambda\)（预算 640）

| Policy | \(R_{0.25}\) | \(R_{0.5}\) | \(R_{0.75}\) |
|--------|--------------|-------------|--------------|
| racing | 0.00469 | 0.00778 | 0.00505 |
| uniform | 0.00580 | 0.00924 | 0.00280 |
| risk_only | 0.00607 | 0.01990 | 0.01542 |
| dice | 0.01062 | 0.01651 | 0.01508 |

在更偏准确率的 \(\lambda=0.75\) 上 Uniform 最好；Racing 在各 \(\lambda\) 均稳健优于 DICE。

### 5.5 H2 消融

Risk-Only 与 Full DICE 宏平均接近（640：0.0138 vs 0.0141），**不能主张 decision impact 带来增益**。

### 5.6 Go / No-Go（针对 DICE 扩容）

依据实施规范 §3：

> 若 Full DICE 在 640 点 Test Preference Regret **不优于** Paired Racing，且 320/1280 无一致性优势，**停止扩容**。

**决定：对 DICE 停止扩容（NO-GO）。** 不扫描超参「寻找胜利」。  
但这**不等于**课程项目结束：固定池离线实验不能替代进化计算集成（见 §6.2）。

### 5.7 三层结论（避免把「假设失败」写成「项目失败」）

| 层次 | 判断 |
|------|------|
| 研究假设（H1/H2） | 当前数据**不支持** DICE / Decision Impact |
| 方法实现 | Racing 与 DICE **混杂了**选对策略与早停/回退机制（见 §6.1） |
| 课程完成度 | Gate 1 不够；需最小进化验证（§6.2），且不覆盖 DICE 负结果 |

准确表述：*在当前固定候选池与注册预算下，Full DICE 未优于 Paired Racing；更广泛有效性未验证；课程所需 EC 集成另做最小验证。*

---

## 6. 机制诊断与进化补课

### 6.1 Task A：预算分配诊断（0 次新 LLM 调用）

脚本：`scripts/diagnose_allocation.py` → `outputs/gate1/diagnosis/`。

主预算 640、20 seed 均值：

| Policy | 预算利用率 | 整列补全占比 | Prompt HHI | Top-3 份额 | 唯一 pair 数 | Regret |
|--------|------------|--------------|------------|------------|--------------|--------|
| racing | **0.995** | **0.145** | 0.154 | 0.593 | 11.3 | **0.0058** |
| uniform | 1.000 | 1.000 | 0.063 | 0.188 | 0 | 0.0059 |
| risk_only | 0.868 | 0.000 | 0.217 | 0.701 | 11.2 | 0.0138 |
| dice | **0.843** | ≈0 | **0.248** | **0.775** | 10.1 | 0.0141 |

要点：

1. **混淆因素（代码已核实）**  
   `PairedRacingPolicy`：`stop_r` + 无可用 pair 时 **回退 Uniform 整列**；  
   `DICEPolicy` / `RiskOnlyPolicy`：`stop_r=None` 且 **无回退**。  
   因此注册对照同时混入「如何选下一对」与「何时停止 / 是否探索」。

2. **DICE 把预算集中在少数 Prompt**  
   HHI 与 top-3 份额均高于 Racing；λ=0.5 赢家中 `p15` 占 **55%**（Racing 仅 20%，更分散于 `p00/p03/p05`）。

3. **Impact 改变了买哪些 pair，但没降低后悔**  
   Jaccard(DICE pairs, Risk-Only pairs)≈**0.22**（集合差异大），但 regret 几乎相同 → 符合「高风险/高影响 ≠ 预期决策损失下降（VOI）」的诊断。

4. **DICE 常吃不满预算**（利用率 84% vs Racing 99.5%），在 pair 查询空转后提前停止。

图：`outputs/gate1/diagnosis/figure_budget_allocation.png`。

**不据此重调 DICE 权重。** 理论后续方向可记为  
\(\mathrm{VOI}(q)=\mathcal{R}(D)-\mathbb{E}[\mathcal{R}(D\cup\{(q,Y_q)\})]\)，作为解释视角而非立刻上线的新模块。

### 6.2 Task B：最小化进化计算验证（不含 DICE）— Table 3

目的：满足课程「进化算法集成」；验证 NSGA-II + Uniform/Racing 相对 Random Search 的可运行搜索。**不**声称 DICE 成功，**不**覆盖 Gate 1 负结果。

| 参数 | 设定 |
|------|------|
| 数据 / 模型 | AG News；Qwen2.5-7B-Instruct |
| 对照 | NSGA-II+Uniform、NSGA-II+Racing、Random Search |
| 种群 / 代数 | 8 / 4 |
| Seeds | 42, 123, 456 |
| 每 run 搜索预算 | 1,280 calls（实际调用约 1256–1280） |
| 变异 / 交叉 | `evolution.py` 规则式算子（已真实执行；多份 winner 为子代 ID） |
| Test | **仅最终 winner** × 200 条密封 test（协议允许的保守设定） |
| 加速 | batch=32；Gate1 矩阵预热初始 8 Prompt；墙钟 **166.9 s** |

#### Table 3a — 逐 seed（winner test）

| Method | Seed | Calls | Test Acc | Test Cost (tok) | Winner |
|--------|------|-------|----------|-----------------|--------|
| nsga2_uniform | 42 | 1280 | 0.765 | 112.3 | `uniform_s42_g3_c03` |
| nsga2_racing | 42 | 1280 | 0.780 | 112.5 | `racing_s42_g3_c02` |
| random_search | 42 | 1280 | 0.800 | 116.6 | `p15` |
| nsga2_uniform | 123 | 1280 | 0.775 | 107.5 | `uniform_s123_g3_c06` |
| nsga2_racing | 123 | 1280 | **0.820** | 122.6 | `racing_s123_g3_c07` |
| random_search | 123 | 1256 | 0.795 | 119.6 | `rand_s123_c25` |
| nsga2_uniform | 456 | 1272 | 0.775 | 103.8 | `uniform_s456_g3_c00` |
| nsga2_racing | 456 | 1262 | 0.775 | 103.8 | `racing_s456_g3_c06` |
| random_search | 456 | 1256 | **0.880** | 115.4 | `rand_s456_c25` |

#### Table 3b — 三 seed 均值

| Method | Mean Test Acc | Mean Test Cost | Mean \(U_{0.5}\)† | Mean Calls |
|--------|---------------|----------------|-------------------|------------|
| NSGA-II + Uniform | 0.772 | **107.9** | 0.175 | 1277 |
| NSGA-II + Racing | 0.792 | 113.0 | 0.175 | 1274 |
| Random Search | **0.825** | 117.2 | **0.184** | 1264 |

† \(U_{0.5}=0.5\,a-0.5\,\tilde{c}\)，\(\tilde{c}=c/256\)（与 Gate 1 同一成本尺度）。

**读法（必须诚实）**：

1. 进化管线**工程上成功**：子代 Prompt 进入最终 winner，预算耗尽规则生效，三方法可复现。  
2. 在 **3 个 seed** 上，Random Search 的平均 test acc / \(U_{0.5}\) 最高；NSGA-II+Racing 的 acc 高于 Uniform，但效用与 Uniform 接近（Racing 成本更高）。  
3. **不能**据此写「NSGA-II 显著优于随机」；也不能把 Random 在 seed 456 的 0.88 外推为稳定优势。  
4. **与 DICE 无关**：本表不是 Gate 1 的补救实验。

原始汇总：`outputs/gate2/online_summary.json`。

### 6.3 Gate 2 公平性审计（0 GPU）— 回答「为什么 Random 更高」

脚本：`scripts/audit_gate2_fairness.py` → `outputs/gate2/audit/`。

#### Test 协议核查（代码）

| 检查项 | 结果 |
|--------|------|
| `evaluate_on_test(..., [winner])` | **2** 处（NSGA / Random） |
| 全种群 test 调用 | **0** |
| 选择依据 | search 上的 \(U_\lambda\) / NSGA 支配排序 |
| Gate1 预热 | **仅 search split** |

结论：未发现用 test 选 Prompt 或调参；报告中的 Test Acc 确为 **winner-only**。

#### 候选生成机会（代码级，不公平）

| 项目 | NSGA-II | Random Search |
|------|---------|---------------|
| 初始 | 8 | 8 |
| 新生成 | **24**（8×3 代，末代不生子代） | **32**（8×4） |
| 设计总 ID | 32 | **40** |
| 算子 | crossover + `mutate_prompt` | **仅** `mutate_prompt` |
| 预算用法 | 按代切片；每代截断到 μ=8 | 生成后 **Uniform 扫整池** |
| 最终赢家池 | 进化截断后的 8 个 | 全池按 \(U_\lambda\) 取 top-8 再选赢家 |

因此：**不能**在同等 footing 上断言「NSGA 选择机制劣于随机」。Random 更接近「多候选 + 一次均匀筛」，探索覆盖更大。

日志侧：NSGA 各 run 进入历史的 child ID 约 8–15（大量子代被代际截断，未进入下一轮 history）；Random 设计池 40，文本去重率约 10–20%（RNG 重放近似）。

#### Search–Test 间隙（NSGA winner）

多个 NSGA winner 的 search_acc 显示为 **1.0**，而 test_acc 约 **0.76–0.82**（gap≈0.18–0.23）。这与「有限预算下过早利用噪声优势」一致，但 **不能**单独证明与 Gate 1 的 DICE 集中化同源——仅为跨实验解释性假说。

---

## 7. Discussion（研究表述）

### 7.0 事后公平消融（Post-hoc；0 新 LLM）★

为回答「是评分机制没用，还是实验不公平」，在**同一冻结矩阵**上离线重放：

| 设定 | 内容 |
|------|------|
| 共享 | `racing_stop_r` 早停 + 无 pair 时 **Uniform 整列回退**；同预算/种子/候选池 |
| 唯一差异 | `use_impact=True`（`fair_dice`）vs `False`（`fair_racing`） |
| 标记 | **exploratory / post-hoc**；**不替代**预注册 NO-GO |

**预算 640（20 seeds）**：

| Policy | Regret mean | Util mean |
|--------|-------------|-----------|
| fair_racing / racing / fair_risk_only | **0.00584** | **0.995** |
| **fair_dice** | **0.01210** | **0.994** |
| 原 dice（不公平实现） | 0.01407 | 0.843 |

- 利用率已对齐（fair_dice ≈ 99.4%）。  
- 配对：`mean(fair_dice − fair_racing)=+0.006`；仅 **20%** seed 上 fair_dice 更好，**70%** Racing 更好。  
- 趋势：320/640 仍明显落后；1280 时接近（0.00677 vs 0.00627）但仍无优势。  
- 相对原 dice，加 scaffold 略降 regret（0.014→0.012），**不足以翻盘**。

**判断**：在公平早停/回退/预算利用之后，**Decision-Critical Scoring 本身仍未显示收益**。  
预注册负结果成立；事后消融把归因从「实现不公平」推进到「打分机制缺乏增益」。  
产物：`outputs/gate1/posthoc_fair/`。

### 7.1 可严格主张的

1. **预注册**：完整原版 DICE **不如**完整 Racing（NO-GO）。  
2. **事后公平消融**：在共享早停+回退后，**Impact 加权仍不优于** risk-only/Racing 脚手架。  
3. 因此：既不是「只因没用满预算才输」，也尚未找到使 DICE 胜出的公平配置。  
4. EC 管线可复现；Gate 2 Random vs NSGA **候选数不公平**，不能据此贬低 NSGA。

### 7.2 解释性假说（Discussion）

> **Decision-aware exploitation is not necessarily information-efficient under noisy, low-budget prompt evaluation.**

公平消融后仍落败，使该假说更指向 **评分目标（风险×影响 ≠ VOI）**，而非仅仅缺 early-stop。仍非跨任务因果证明。

### 7.3 课程创新表述（必须诚实）

严谨负结果 + 机制审计 + 事后公平消融，体现研究完整性；**不等于**有效算法创新。可写清：提出了什么、如何预注册、事后如何追问归因、结论边界在哪里。

---

## 8. 局限

- 单数据集、单模型；Gate 1 固定 16 Prompt。
- 事后消融不可替代预注册；不可写成「第二次主实验胜利/失败洗白」。
- Gate 2 候选数/算子仍未公平重跑（有意停扩）。
- EC 仅 3 seeds；Related Work 需 ≥20 篇主题综述。

---

## 9. 可复现材料

```bash
cd /home3/river/DSAI6707/DICE_PO
.conda/bin/python scripts/run_posthoc_fair_ablation.py   # 0 new LLM
.conda/bin/python scripts/audit_gate2_fairness.py
.conda/bin/python scripts/diagnose_allocation.py
```

| 路径 | 说明 |
|------|------|
| `outputs/gate1/GO_NOGO.md` | 预注册 NO-GO |
| `outputs/gate1/posthoc_fair/` | 事后公平消融 |
| `outputs/gate1/diagnosis/` | 机制诊断 |
| `outputs/gate2/online_summary.json` | EC Table 3 |
| `outputs/gate2/audit/` | Gate2 公平性审计 |

---

## 10. 结论

1. **预注册负结果保留**：原版 DICE 未优于 Racing。  
2. **公平消融后仍负**：对齐早停/回退/利用率后，Impact scoring 仍无优势 → 更支持「机制本身缺收益」。  
3. **项目未白做**：可复现栈、诊断、事后归因、EC 集成均已交付。  
4. **停止继续改 DICE 找胜利**；转文献综述与 Presentation。可选未来工作：VOI 型查询价值（需新预注册）。

---

## 11. 6 分钟展示提纲（草稿）

| 时段 | 内容 |
|------|------|
| 0:00–0:45 | 问题与预算 |
| 0:45–1:30 | DICE 主张 |
| 1:30–3:00 | 预注册负结果 |
| 3:00–4:15 | 不公平点 → 事后公平消融仍负 |
| 4:15–5:15 | EC 跑通 + 候选数不公平审计 |
| 5:15–6:00 | 负结果的研究价值与边界 |

---

## 参考文献（书目需在终稿逐条复核并扩至 ≥20 篇）

1. MO-CAPO (2026). https://arxiv.org/html/2605.18869v1  
2. BudgetAPO (2026). https://arxiv.org/html/2610.05671v2  
3. CAPO (2025). https://arxiv.org/abs/2504.16005  
4. TRIPLE (2024). https://arxiv.org/abs/2402.09723  
5. IPOMP (2025). https://arxiv.org/abs/2505.10736  
6. SESS (2026). https://arxiv.org/html/2601.03493v1  
7. GEPA (2025/2026). https://arxiv.org/abs/2507.19457  
8. RiOT (2025). https://aclanthology.org/2025.acl-long.1086/  

（终稿 Related Work 需按「方法—局限—缺口」主题组织，并补足至课程要求的 2024–2026 文献数量。）

---

*本报告 Gate 1 数据可由 `outputs/gate1/` 重算；EC 数据见 `outputs/gate2/online_summary.json`。未编造显著性，未将 EC 结果表述为 DICE 成功。*
