**184 项按现有声明范围分为：50 项软件浮点位语义证明、133 项依赖后端算子的证明、1 项仅修复候选。三类均不表示 GPU 已验证。**

本轮保留原 747 条公开声明，新增 125 条：121 条数值证明（其中 6 条针对独立模型）、2 条未写内存保持不变的证明、1 条递推恒等式、1 条协议证明；合计 872 条。固定样例不计入参数化证明。

这些数字按已接受的输入、配置和阶段范围统计，不表示所有 Python dtype、host 分支或硬件行为已覆盖。50 项证明对应声明中的软件浮点模型和配置；133 项证明仍把明确列出的后端算子当作参数，需要另外建立这些算子与实际硬件的对应。部分旧证明描述源程序与候选程序的相对等价。LayerNorm、spinning lock 和 StreamK 的任务局部顺序一致（SC）模型单列，不算共享 DSL 或 GPU 并发正确性。

原始 FP4 scaled 源仍缺可验收的原始源码证明，当前只有编译修复候选。SGMV 的逗号修复候选与原始源分别记录，候选编译成功不改变原始编译失败记录。

此前 41 项修复中，30 项是软件浮点位语义证明，11 项是依赖后端算子的证明。其 118 条声明已包含在原 747 条中，不再与本轮新增 125 条相加；逐条名称、模型、statement SHA 和当前审核源码 SHA 见 [原 41 项绑定](original41-bindings.json)。原已接受范围与剩余范围原样保留在 TSV。

原 41 项的 native 对应事项、37 个实验工作项、原 5 个精确配置相互有重叠，不能相加当作独立任务总数。

实际机械检查状态：

| 检查 | 状态 |
|---|---|
| source_inventory | passed |
| module_registration | passed |
| reviewed_source_bindings | passed |
| original_source_presence | failed |
| python_source_bindings | passed |
| integration_build | passed |
| axiom_audit | passed |
| comparator_preflight | failed |
| corpus_replay | not_run |
| legacy_source_audit | passed |
| library_replay | not_run |
| value_metadata | passed |
| supplemental_ci_24 | passed |

官方 comparator 隔离预检失败时，library/corpus replay 保留 `not_run`。API 的统一门禁因此仍未全通过；已通过检查和未运行项目分别列出。

数值实验见[37 项实验交接和原 5 项精确配置](evidence/native-handoff/EXPERIMENT-HANDOFF.md)。37 项都尚未交付完整 GPU 采集器与独立批量 expected evaluator；原 5 项配置不被它们替代。已有 111 条离线编译记录为 91 成功、20 失败，均非 GPU 数值执行。正常输入与已接受阶段限制不重新计作实验债务。

已知反例（全域 fp16 舍入恒等、任意 dot+add 与 dotAcc/FMA 等价、NaN 域 compare/select 与 maximum 等价等）不列成等待实验验证为真的假设。

逐项范围、配置、公开声明及实验关联见 [184 项 TSV](tasks-184.tsv)。
