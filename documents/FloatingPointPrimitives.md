# 浮点执行原语与证明接口

VeriTile 分别提供实数数学正确性、选定软件模型内的严格浮点定理，以及依赖实验规则的
实现等价推导。实际源码由 `triton { ... }` 保存；Python/Triton 执行基础表达式的数值实验，
Lean 检查定理条件、源码连接与组合证明。

## 源码、执行与规格

| 层次 | 代码入口 | 职责 |
|---|---|---|
| 源程序 | `Kernels.lean`、[Compute AST](../VeriTile/Triton/Core/Ast.lean) | 保存操作顺序、精度、cast、mask、分支和存储 |
| 数学正确性 | `RealCorrect.lean`、[Spec.Real](../VeriTile/Spec.lean) | 在实数解释下证明独立数学公式及相应执行、内存和输出条件 |
| 浮点执行视图 | [Structural](../VeriTile/Triton/Float/Structural.lean)、[ExecutionProfile](../VeriTile/Triton/Float/ExecutionProfile.lean) | 指定数值操作的解释与默认精度，保留显式精度和转换 |
| 局部关系 | [GuardedRules](../VeriTile/Triton/Float/GuardedRules.lean) 及各原子库 | 记录左右片段、操作数域、精度、intrinsic 与可用证据 |
| 程序等价 | `FPEquiv.lean`、[FP Kernel](../VeriTile/Triton/Float/Kernel.lean) | 将局部规则接到实际程序的执行、输出、调度与 memory frame |
| 数值测量 | [two-gates runner](../scripts/check_two_gates.py) | 执行原子表达式对，检查两道门并生成可独立 CPU 重放的证据 |

`IO.lean` 保存共享内存布局、输入输出绑定和调度，`Contract.lean` 保存输入约束与
数学规格。两个公开证明入口各自使用这些定义；具体组织见 [示例索引](../bench/examples/README.md)。

实数解释按其数学定义处理算术和转换。FP 证明保留源精度、中间转换、FMA 与分开乘加、
归约顺序和 dot 参数。显式标记的精度不会被默认配置覆盖。执行失败、未定义 lane 和
越界访问由执行与安全判断处理；mask 的 `other` 值仍由实际源码决定。

归约、scan、dot、完整 softmax、Welford 和 LayerNorm 是组合证明的对象。
它们的顺序、分块和累加计划须连接到对应源码；局部原子的测量结果不直接给出完整算法的结论。

## 软件浮点值与严格证书

[BitValue.lean](../VeriTile/Triton/Float/BitValue.lean) 定义 bf16/fp32 的
`Value format`，以格式对应的 `BitVec` 保存全部位模式。解码保留符号、有限有理数值、
无穷、NaN 的 signaling 位与 payload。`Config` 明确选择舍入模式以及输入、输出的
subnormal flush；`Config.ieeeValue` 使用 nearest-even 且两端均不 flush。

[ScalarOps.lean](../VeriTile/Triton/Float/ScalarOps.lean) 实现逐结果舍入的
add/sub/mul/div、一次舍入的 FMA，以及数值相等与有序比较。混合精度通过显式转换表达。
算术与转换的 NaN 策略是 canonical quiet NaN；原始位模式仍可区分 payload 和带符号零。
软件模型不观测异常 flags。数值比较、位相等和有限实数投影各有自己的定义域。

严格模型证明按实际源片段及其 typed load/rnd/store 路径提供证书：

| 库 | 范围 |
|---|---|
| [ScalarProbeExact](../VeriTile/Triton/Float/ScalarProbeExact.lean) | 指定精度和操作数域的标量关系 |
| [MaximumProbeExact](../VeriTile/Triton/Float/MaximumProbeExact.lean) | 指定 maxnum 语义及存储转换的关系 |
| [PrecisionProbeExact](../VeriTile/Triton/Float/PrecisionProbeExact.lean) | 具体转换路径与源码片段的精度恒等关系 |
| [ExponentialProbeExact](../VeriTile/Triton/Float/ExponentialProbeExact.lean) | 各探针精度下指数特殊值的执行证书 |
| [CountConversion](../VeriTile/Triton/Float/CountConversion.lean) | int32 零与有界后继到 FP32 的转换 |

[CurrentAdmission](../VeriTile/Triton/Float/CurrentAdmission.lean) 将完整精度、源片段和
操作数域与严格证书或实验记录匹配。严格条目实际携带证明；定理名称字符串不能构造证书。
指数模型的 ABI、特殊值与残余算法参数见 [特殊值说明](FloatingPointSpecialValues.md)。
严格证书只适用于其选定软件模型与前提；GPU 对该模型的符合性是单独的连接义务。

[Counterexamples.lean](../VeriTile/Triton/Float/Counterexamples.lean) 给出六个由 Lean 内核
计算检查的 bf16/fp32 不等见证，涵盖加法结合重排、分配和 FMA。
它们说明这些关系不能一般地作为严格浮点恒等式，具体输入和结果见
[规则指南](FloatingPointRewriteRules.md#3-已计算的严格不等见证)。

## 实验原子及证明依赖

原子库分别定义算术、倒数、指数、log、maximum 和整数转换关系：
[ScalarArithmetic](../VeriTile/Triton/Float/ScalarArithmetic.lean)、
[Reciprocal](../VeriTile/Triton/Float/Reciprocal.lean)、
[Exponential](../VeriTile/Triton/Float/Exponential.lean)、
[LogExp](../VeriTile/Triton/Float/LogExp.lean)、
[Maximum](../VeriTile/Triton/Float/Maximum.lean) 和
[CountConversion](../VeriTile/Triton/Float/CountConversion.lean)。
候选表达式的定义独立于它是否得到实验准入。

`GuardedFragment` 保存 `guards`、实际 `ComputeStmt` 序列、标量通道 dtype 和可选的
默认运算精度。有限、非零、正值等条件在规则使用处检查；若操作数来自前缀执行，
条件针对该位置的真实寄存器值。输入叶节点有限不能代替中间操作数的有限性证明。
`tl.exp`、`libdevice.exp`、普通除法和 `div_rn` 保留不同身份。

[Spec.lean](../VeriTile/Spec.lean) 的接口分为：

- `AtomicRule` 和 `RuleEntry`：左右片段、冻结配置、规则身份与证据记录。
- `AcceptedAtom`：`EvidenceValidated` 外部验证前提、实例匹配、有效 artifact 和允许的双门结果。
  `AcceptedAssumptions` 对规则表逐项要求这些义务。PASS 字符串本身不能构造证据。
- `Derivation` 和 `FloatingPoint`：从固定严格理论和实验假设组合推导，并保持程序签名。
  IO 视图还保留其定义的完整执行、输出与 frame 条件。

公开接口为 `Spec.Real claim` 和 `original ≡[R] candidate`。
`R` 记录实验假设；空 `R` 仍可使用程序视图固定的严格理论。
对称、传递与共同上下文是形式推导规则；有向数值检查不因此获得反向或组合的统计保证。

`#print_fp_assumptions theorem_name` 遍历实际证明及实例化的可达辅助证明，分别打印严格证书
和引用的实验原子。同一实例去重，不同配置分开；不可展开的 FP 前提标为 unresolved。
该输出不声称给出逻辑上最小依赖集。`#print_spec theorem_name full` 可检查配置与证据细节，
公理审计和 comparator 则独立检查其各自的义务。

## 数值运行与证据接口

默认命令和配置如下；完整参数与重放要求见 [实验指南](../experiments/floating_point/README.md)。

```bash
python3 scripts/check_two_gates.py check
python3 scripts/check_two_gates.py run --smoke --output Logs/two-gates-smoke
python3 scripts/check_two_gates.py run --output Logs/two-gates
python3 scripts/check_two_gates.py import Logs/two-gates --output Logs/two-gates-replayed.json
```

[共享配置](../experiments/floating_point/two_gates_config.py) 控制预算、执行形状、种子、并发
和门槛。每个可运行配置固定运行一次，原分布峰值输入预算为 `2**32` 组。
关系配置提供分布、精度、原子选择与 launch；执行形状只限制工作和内存。

主目录 FP32 的 CANCEL、ADD-ASSOC、MUL-DISTRIB 在独立 Normal(1,1) 下共用
`2**25` 组近零提议与 `2**32` 组补集条件输入。
均值使用全库分层权重；峰值使用独立二项式层计数、接受—拒绝和固定前缀恢复的
原分布样本多重集，并对两侧误差分别取未加权最大值。其他配置按声明分布直接采样，权重为一。

均值门要求 `abs(mean)+5*SE <= 0.05 ULP`，峰值门要求
`candidate_peak <= 10*reference_peak`。完整判决、输入域、非有限筛选、供给不足和证据边界见
[接受契约](TwoGatesAcceptance.md)。预算及选择方式在观察输出前冻结，运行固定预算后给出结果。

manifest 绑定配置、源码快照、随机流、软件/设备与编译身份。原始 bundle 保存全部执行分块的
统计、输入输出指纹、PTX 和必需的原始审计。独立 CPU 导入核对这些证据并重算最终判决。
[验证回执](../experiments/floating_point/validation/README.md) 记录全量结果及对应源码散列。

数值报告与 Lean 准入表分别维护。现有导出器检查所支持报告格式的完整性，并生成规则数据；
实际证明还要求片段、精度、intrinsic、操作数域与外部证据前提对应。
当前 runner 的报告不自动写入 Lean 规则表，也不自动启动证明 agent。
`scripts/prove.sh` 提供单任务证明入口；目标、规则与可信文件需在证明前固定，Lean/comparator
检查所生成的证明。这个显式流程的运行边界见 [源码与证明示例](../experiments/floating_point/EXAMPLE.md)。

实验尺寸与分布用于选择原子假设；后续符号化 kernel 的尺寸可以独立变化。
局部抽样分布与实际中间操作数分布的统计联系需要单独论证。
有限样本的数值接受是规定配置下的证据，不提供全域 IEEE 位值相等或整个 kernel 的误差界。
实际规则用量、公开条件和延期证据见 [证明依赖指南](FPRemainingAdmissionGaps.md)。
