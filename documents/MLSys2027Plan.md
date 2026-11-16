# VeriTile MLSys 2027 投稿准备

本文记录论文主张、证据范围与需要独立完成的评测。实现边界以
[语义说明](SemanticCaveats.md)、[规格接口](CorrectnessSurfaces.md) 和
[数值接受契约](TwoGatesAcceptance.md) 为准；任务与示例的选择见
[源码验收指南](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/documents/bench/current_source_acceptance.md)。
长期路线见 [PLAN.md](../PLAN.md)。

## 方法与可支持的主张

VeriTile 是用于 ML kernel 的验证框架。论文需要分别展示两条证明路线：

- `RealCorrect.lean` 在实数语义下证明源程序实现独立数学公式，保留适用的输入、索引、
  内存、调度与输出条件。
- `FPEquiv.lean` 保留源精度、cast、FMA 区别和计算顺序，使用固定软件模型内的严格证书
  及显式实验原子，证明 `original ≡[R] candidate` 并报告实际依赖。

Two-gates 测量固定精度、后端和分布下的局部表达式对。均值门检查带符号的 local-ULP
偏差，峰值门比较双方相对 oracle 的最大绝对误差。实验只为对应原子提供数值证据；
完整 softmax、Welford、LayerNorm、归约和内存性质通过 Lean 的组合证明取得。
数值接受不证明全域 IEEE 位值相等，也不自动给出整个 kernel 的误差界。

论文应以实际源码、完整公开条件和可重放证据支持案例主张。GPU 与软件规格模型的符合性、
所有执行输入上的位值正确性、配置到证明的全自动流水线，均需要各自独立的证据。

## 实现与数值实验

[原语与证明接口](FloatingPointPrimitives.md) 说明值语义、typed 执行视图、局部关系与程序连接。
[规则指南](FloatingPointRewriteRules.md) 给出候选公式、条件与六个严格不等见证。
[规则使用指南](FPRemainingAdmissionGaps.md) 区分可用目录、实际主定理依赖和延期数值证据。

数值入口是 [check_two_gates.py](../scripts/check_two_gates.py)，统一设置位于
[two_gates_config.py](../experiments/floating_point/two_gates_config.py)。每个可运行配置
固定运行一次，原分布峰值输入预算为 `2**32` 组。主目录三条指定 FP32 关系用额外
`2**25` 组近零提议组成共享样本库：均值使用全库分层权重，峰值使用恢复原分布的
共同选择集上的未加权误差。其他配置使用其声明分布和单位权重。

运行、resume、独立 CPU 导入与证据格式见 [实验指南](../experiments/floating_point/README.md)。
[验证回执](../experiments/floating_point/validation/README.md) 绑定全量测量源码和结果。
论文引用结果时必须匹配回执的精度、输入域、预算、源码与后端。
数值拒绝、不支持、证据不足与执行错误各自报告。

Lean 准入表与实验报告分别维护。导出规则数据和证明使用仍需检查对应的报告格式、
具体片段、精度、intrinsic、操作数域和外部证据前提；runner 不自动写回准入表或启动 agent。
`scripts/prove.sh` 提供单任务证明入口，Lean 和 comparator 独立验收生成的证明。

## 案例与证明范围

每个论文案例应列出以下对象，并能从公开定理追溯到实际源码：

| 对象 | 论文需说明的内容 |
|---|---|
| 输入与数学公式 | 逻辑尺寸、独立目标、允许域与完整输出范围 |
| 源程序与候选 | 逐句对应的 Python/Lean 函数体、转换、mask、分支与调用顺序 |
| Real 证明 | 成功执行、源访问安全、输出公式与 memory frame |
| FP 证明 | 比较的实际程序对、精度、局部关系、归约/调度计划及其适用条件 |
| 依赖 | `#print_fp_assumptions` 的严格证书、实验原子与 unresolved 前提 |
| 验收 | 源码对照、Lean 构建、公理审计、statement 检查和 comparator 重放 |

参数化抽象舍入模型的结果要标明其算术与舍入点。
[RoundingModel.lean](../VeriTile/Triton/Float/RoundingModel.lean) 规定同 dtype 的幂等性；
单调性、奇对称性和跨精度可表示关系按各定理的实际前提使用。
实数代数事实、抽象 cast/store 舍入结论和逐操作浮点定理分别报告。
例如 fused SwiGLU 的中间量化、各 stage body 的顺序组合和独立 launch 的调度连接，
须按公开定理实际覆盖的范围解释。

## 评测与复现

以下评测需各自有固定入口、冻结配置和独立结果，不能从构建成功或数值门通过推定：

1. **证明覆盖。** 按公开判断计数源码对齐、Real/FP 定理、辅助定理与延期证据，保留全部条件。
2. **验证成本。** 记录源码对照、Lean 构建和 comparator 的墙钟时间、CPU/内存与缓存状态；
   批量审计须与完整定理清单一致。
3. **复用收益。** 比较共享 IO、共享局部引理和题目专属证明的职责与用量，说明新增案例需要的工作。
4. **数值成本。** 按规则、精度和输入域报告固定预算、均值、标准误、两侧峰值、判决、耗时和显存。
5. **失败分析。** 区分逻辑反例、规则不适用、数值拒绝、未支持配置、执行错误与证明未完成。
6. **相关工作比较。** 固定可比较的任务、语义范围和预算，用实际运行支持能力及成本主张。

复现材料固定 VeriTile/VeriTileBench 提交、Lean/Python/Triton 环境、编译设置、输入种子、
定理清单和测量 bundle。紧凑回执随源码保存，原始数据与完整重放证据单独归档。
CPU CI 检查轻量回归；GPU 数值实验显式运行。自动证明成功率需要独立固定任务与预算的实验。
