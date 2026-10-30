# 浮点变换规则与接受结果表

更新日期：2026-10-21。当前目录有 **42 个严格模型实例和 36 个保留实验实例**；183 个任务主声明与 21 个示例 FP 主规格合计实际使用其中 **12 个实验实例**。完整主用清单与条件见[当前规则使用指南](./FPRemainingAdmissionGaps.md)。原始 14 条候选关系的 42 个格式检查仍保存在[历史实测结果表](../experiments/floating_point/report/summary.md)。第 3 节保留六个经 Lean 内核检查的严格不等反例；它们否定通用严格相等，不等于 two-gates 拒绝。

这里的“等价性原语”对应可复用的变换规则，只包括固定规模的局部关系，例如结合律、分配律、FMA 与具体 cast 关系。归约重排和完整算法变换是待推导结论，不是准入原子。它与 [浮点执行原语](./FloatingPointPrimitives.md) 分开：执行原语定义程序怎样计算，本表记录参考计算如何变成候选计算，以及允许该变换的证据。

## 1. 表的单位与判决

**候选目录一行是一条有向规则；结果表一行是该规则在一个完整配置下的检查。** 同一条规则可以有多个结果，不能只按名字登记一次 PASS。

结果键至少包含：

    规则 ID + 参考/候选执行图身份
    + 输入/操作/累加/输出精度及 rounding/intrinsic 配置
    + 逐元素测试张量的 shape/stride 和 launch（归约、scan、dot 计划为 null）
    + 后端与双方编译/启动配置
    + 输入域/权重/高斯探针配置
    + two-gates 协议与检查器版本

分别保存严格等价证据和统计接受证据：

| 字段 | 可记录的值 | 含义 |
|---|---|---|
| 代数证明 | theorem + 前提 / 待证明 / 反例 | 理想算法层的性质 |
| 严格浮点关系 | theorem + 前提 / 未证明 / 数值反例 | 选定浮点语义下的严格关系 |
| Bias gate | NOT_RUN / PASS / FAIL / INCONCLUSIVE | 局部 ULP 单位的平均偏差预算 |
| Vars gate | NOT_RUN / PASS / WARN / FAIL / INCONCLUSIVE | 当前配置的误差放大与尾部检查 |
| 接受决定 | NOT_EVALUATED / ACCEPT / ACCEPT_WITH_WARNING / REJECT / INCONCLUSIVE | 按契约合并证据与两门判决 |

匹配源码精度、转换、存储、模型和输入域的严格证书可直接用于证明，并移除对应的当前实验义务。其余实验准入条目须有配置匹配的两门证据；未运行的 gate 保持 NOT_RUN。历史报告保留，严格证书不伪装成实验结果。纯语法相同可用自反规则。

对外正确性使用实数语义 `Spec.Real`，实现等价性沿用 `lhs ≡[R] rhs`（`VeriTile.Spec` scope）。`R` 记录实验准入规则；程序视图还可提供固定的严格软件理论。`#print_fp_assumptions` 分开显示实际使用的实验规则和严格证书，不把整个目录视为已用依赖。依赖 `R` 的形式组合不构成无条件 IEEE 位值相等或整 kernel 误差界。

统计接受路径按 [two-gates 协议](./TwoGatesAcceptance.md) 合并结果。待证明、待测试和证据不足分别记录，不能当作已证明错误；相反，某个输入上的数值不等也不能自动推导统计 FAIL。

## 2. 候选规则目录

以下公式是模式说明，正式检查必须实例化为完整的 typed Compute IR，明确每一步精度、转换和计算顺序。\(q_d\) 表示指定配置下转换到格式 \(d\)；数学规格会按其定义处理或投影这些转换。箭头标注数值检查的参考到候选方向；反向的实验结论需要另建记录。准入后的形式等价假设可以使用对称规则，但这不产生反向实验记录。

候选关系先在 Lean 中定义左右片段和条件，定义本身不依赖是否通过实验。
[LogExp.lean](../VeriTile/Triton/Float/LogExp.lean) defines twelve FP32 candidates
using independent backend parameters:

| Lean constructor | Rewrite |
|---|---|
| `log_mul log` | Unconditionally split `log(a*b)`; log selects `.tl` or `.libdevice` |
| `log_mul_split log` | Keep the product log in `[0.5, 2]`, otherwise split; each log is independently admitted |
| `log_exp log exp` | Unconditionally eliminate `log(exp(a))`; four log/exp combinations |
| `log_exp_cancel log exp` | Eliminate only for `0.5 < abs(a) <= 80`, otherwise retain the reference; four combinations |

The exp argument of `log_exp_cancel` defaults to `.libdevice`. Product relations
require finite positive inputs; log-exp accepts finite signed inputs. Each
backend combination has separate typed fragments and report IDs; no API equality
is assumed. Both FMA/log1p diagnostics remain experiment-only. The default
assumption printer lists only used atoms, such as `log_mul_split(tl.log)` or
`log_exp_cancel(tl.log, libdevice.exp)`.

two-gates 结果生成 [LogAdmission.lean](../VeriTile/Triton/Float/LogAdmission.lean)，
`Atom.report?` 按规则、精度和定义域选择已准入条目，`Rules.assumptions` 自动收集这些条目。
未准入的候选仍可用于表达目标或编写条件证明，候选库在准入表为空时也能编译。
已准入后，通过统一的 `rewrite` 引理使用：

```lean
import VeriTile.Triton.Float.LogExp

open VeriTile.Triton.FP.LogExp
open scoped VeriTile.Spec

example (R : Rules) :
    [(Atom.log_exp_cancel .libdevice).lhs] ≡[R] [(Atom.log_exp_cancel .libdevice).rhs] :=
  VeriTile.Triton.FP.LogExp.rewrite R (.log_exp_cancel .libdevice) (by decide)
```

这里 `by decide` 只检查当前表是否选中了该候选；数值关系仍是 `R` 中的外部准入假设。
刷新实验结果后，已定义候选的可用性随准入表更新，无须重写候选表达式。
`log_exp_cancel .libdevice` 对应 `LOG-EXP-GUARDED`，原始表达式始终是
`libdevice.log(libdevice.exp(a))`。候选只在 `0.5 < abs(a) <= 80` 时返回 `a`，
其余输入保留原计算；未选中的 fallback 参数先置零。两轮种子的 fp32 实验均通过。
分支属于候选程序，输入条件仍是 finite，不要求输入落在消去区间。
该关系不能用于无条件消去 log-exp。完整例子的公开规格是
[`log_exp_equiv`](../bench/examples/LogExp/FPEquiv.lean)。

条件式乘积关系 `log_mul_split .libdevice` 默认打印为
`log_mul_split(libdevice.log)`；它对应已冻结的
实验标识 `LOG-MUL-GUARDED`。在
`0.5 <= fp32(a*b) <= 2` 时保留 `libdevice.log` 的乘积形式，范围外才拆分。
左右候选保留 fp32 精度、正有限输入以及未选分支传入 `1` 的处理。

```lean
example (R : Rules) :
    [(Atom.log_mul_split .libdevice).lhs] ≡[R] [(Atom.log_mul_split .libdevice).rhs] :=
  VeriTile.Triton.FP.LogExp.rewrite R (.log_mul_split .libdevice) (by decide)
```

`apply_log_mul_split` 将该原子用于具体的正有限操作数，并保留比较器支持和
乘积分支；它不提供无条件拆分。`log_mul .tl` 和 `log_mul .libdevice` 仍未准入。
尚未准入的 FMA/log1p 实验仍只保留实验定义，接入它需要显式融合运算语义。
数值结果和独立种子复核见[补充实验](../experiments/floating_point/supplement/README.md)。

其他原语库同样显式列出 `Atom`、`ruleID`、公式、intrinsic、精度与定义域，
定义左右片段，再分别连接严格证书或实验准入证据。

| 候选库 | 预定义的关系 |
|---|---|
| [ScalarArithmetic](../VeriTile/Triton/Float/ScalarArithmetic.lean) | fp32 加乘交换、结合、分配、消去，以及零、一和倒数相关的 11 条关系 |
| [Reciprocal](../VeriTile/Triton/Float/Reciprocal.lean) | 普通 Triton 除法改写为乘倒数；分别保留 fp32 和 fp64 计算后转 fp32 的片段 |
| [Exponential](../VeriTile/Triton/Float/Exponential.lean) | `exp_sub exp`、`exp_zero exp`、`exp_neg_inf_sub exp`；每条分别定义 `.tl` 和 `.libdevice` 两个候选 |
| [Maximum](../VeriTile/Triton/Float/Maximum.lean) | fp32 maximum 的交换、结合、幂等和负无穷单位元 |
| [CountConversion](../VeriTile/Triton/Float/CountConversion.lean) | int32 到 fp32 的零转换，以及 `0 ≤ i < 2^24` 上的后继转换 |

`EXP-SUB-INTRINSIC` 仍是未准入候选，不能用已通过的 libdevice 结果启用它。
这六个 fp32 候选中，零与负无穷特殊值的四个实例已接入严格证书；
`exp_sub .libdevice` 使用实验准入，`exp_sub .tl` 未准入。
[ExpAdmission.lean](../VeriTile/Triton/Float/ExpAdmission.lean) 保留原实验记录。
候选的 exp 参数默认 `.tl`；`exp_sub .libdevice` 必须显式选择。
打印器将 `exp_sub(libdevice.exp)` 列为实验依赖，将已用的特殊值定理列为严格证书。
两种实现的证据不会合并，也不产生 `tl.exp = libdevice.exp` 的桥接假设。

程序优先保留 `tl.exp`。SoftmaxReciprocal 和 FloatDTypeSoftmax 仅改写除法，
共同前缀中的 exp 按原样解释，因此都使用 `tl.exp`，不依赖 exp 原子。
SoftmaxStable、OnlineSoftmax、StableLogSumExp 的 exp-sub，以及 LogExp 的条件消去，
需要当前仅由 libdevice 通过的原子，因此在相应代码和证明中显式使用 `libdevice.exp`。
这按各个证明所需的关系选择实现，不把某一条关系失败推广为整个 API 不可用。

计数后继规则的整数上界属于关系定义域；报告采用不同范围时，不能启用这里的固定范围候选。
它与实验 shape 无关。库不会把 bf16 输出或 fp64 计算的条目当成裸 fp32 关系。

`ScalarArithmeticLaws.lean`、`ExponentialLaws.lean` 和 `CountConversionLaws.lean`
保存局部严格定理与当前已选实验原子的组合推导。它们在实际引用处检查所需证据，
不会把目录中其他已准入候选一并算作证明依赖。


### 2.1 已有严格证书的规则

下列原目录实例已有匹配的模型证明；适用范围包含其 typed load/rnd/store 路径。
这不是所有硬件实现的一致性声明。全部 42 个严格实例及模型边界见[当前规则分类](../VeriTile/Triton/Float/CurrentAdmission.lean)。

| ID | 规则 | 模式 | 关键前提与当前证据 |
|---|---|---|---|
| ADD-COMMUTE | 加法交换 | \(a+b\to b+a\) | 三个目录精度配置已有严格证书；保留配置、转换与原有限域 |
| MUL-COMMUTE | 乘法交换 | \(ab\to ba\) | 三个目录精度配置已有严格证书；观测精度与特殊值模型明确 |
| ROUND-IDEM | 同格式重复舍入消除 | \(q_d(q_d(x))\to q_d(x)\) | 三个目录配置已有转换／存储证书；不推广到任意 cast 或 flush 配置 |
| BF16-WIDEN-RETURN | 提升后转回 | bf16 → fp32 → bf16 | 两个 BF16 输入配置已有 preserving conversion 证书 |

### 2.2 需要按配置判定的数值变换

本节记录候选公式。匹配严格证书的实例使用定理；其他实例必须保留匹配的实验准入。历史 gate 判决见[结果表](../experiments/floating_point/report/summary.md)，当前分类以 `CurrentAdmission` 为准。

| ID | 规则 | 参考 → 候选 | 要绑定的条件或主要差异 |
|---|---|---|---|
| ADD-ASSOC | 加法结合重排 | \((a+b)+c\to a+(b+c)\) | 每次加法精度；不同角色的均值/尺度；第 3 节有严格不等反例 |
| MUL-ASSOC | 乘法结合重排 | \((ab)c\to a(bc)\) | 操作精度、尺度、overflow/underflow |
| MUL-DISTRIB | 乘法分配 | \(a(b+c)\to ab+ac\) | 舍入次数与 FMA 选择；第 3 节有严格不等反例 |
| FMA-CONTRACT | 乘加融合 | add(mul(a,b),c) → fma(a,b,c) | 分离乘加与一次舍入；第 3 节有严格不等反例 |
| CANCEL | 减加消除 | \((a-b)+b\to a\) | 抵消、运算格式、特殊值 |
| DIV-RCP | 除法改乘倒数 | `tl.div_rn(a,b) → a * tl.div_rn(1,b)` | 绑定实际 intrinsic 和额外乘法舍入；不能据此准入普通 Triton `/` 的替换 |
| SQRT-RSQRT | 倒平方根替换 | \(1/\sqrt{x}\to\operatorname{rsqrt}(x)\) | 理想规格要求 \(x>0\)；绑定函数实现及输入域 |
| CAST-MOVE | 输入量化后移 | \(q_d(q_d(a)+q_d(b))\to q_d(a+b)\) | 两个 BF16 输入配置已有严格证书；保留中间加法格式、提升路径和同一份输入 |
| CAST-REMOVE | 移除一个中间 cast | `q_compute(q_bf16(a+b) * c) → q_compute((a+b) * c)` | 两边使用相同输出 cast；只检查这个局部表达式，不量化任意 F/G |
| ACC-WIDEN | 三项和的局部精度提升 | `(a+b)+c` 的输入格式中间结果 → fp32 中间结果 | FP32 输入实例已有严格证书，两个 BF16 输入实例保留实验准入；不代表整个归约或 dot 的累加器可以直接替换 |

DIV-RCP、SQRT-RSQRT 等具有定义域前提的规则，需要明确从探针到合法操作数的生成方式，例如 kernel 内部生成正的平方和。条件化/变换后的分布必须登记，不能称作未经修改的标准高斯。所有规则都要处理特殊值，不得在查看差异后静默筛掉坏样本。

### 2.3 不进入原子假设表的内容

以下旧条目已从目录、GPU 执行入口和结果导入允许列表删除：

- 完整算法：`SOFTMAX-SHIFT`、`SOFTMAX-ONLINE`、`LAYERNORM-WELFORD`、`SWIGLU-FUSE`。
- 组合计算：`REDUCE-REORDER`、`REDUCE-SPLIT`、`SCAN-REORDER`、`DOT-LOWER`、`DOT-ACC-FUSE`、`GEMM-SPLIT-K`。
- 结构和内存性质：`LAYOUT-INVERSE`、`STORE-LOAD-FORWARD`。

前两组由 Lean 根据局部严格定理、已准入关系和结构性推导证明；后一组需要索引、别名与内存条件的证明。整 kernel 数值实验即使通过，也不能将这些结论直接加入 `R`。

19 组示例已完成各自声明的 FP 证明；其[实际规则用量与适用边界](./FPRemainingAdmissionGaps.md)分别记录严格证书、实验依赖和未准入替换。旧主表的反模型不能据此记作当前组合理论的缺口。仍须保留非零分母、具体 intrinsic、精度和 guarded fallback 等条件。

`shape = [4096, 4096]` 表示对原子表达式进行逐元素批量测试，不把该表达式升级为整张量归约假设。

已有证明只作为对应语义层的基础：

- [RoundingModel.lean](../VeriTile/Triton/Float/RoundingModel.lean) 给出抽象幂等性约束及 cast/store 引理。
- [FusedSwigluEquiv.lean](../bench/examples/FusedSwiglu/Proofs/LegacyRealEquiv.lean) 保留中间舍入的抽象模型结论；当前公开证明的范围仍是 stage bodies 的 `ComputeKernel.seq`，不声明独立 launch 的组合性质。
- [FusedLayerNormEquiv.lean](../bench/examples/FusedLayerNorm/Proofs/LegacyRealEquiv.lean) 有实数中间计算及输出抽象舍入的等价性结果。
- [OnlineSoftmaxCorrect.lean](../bench/examples/OnlineSoftmax/Correct.lean) 证明实际两遍 online 源码的输出、内存边界与 frame；其 FP 关系由独立文件证明。

## 3. 已计算的严格不等见证

以下输入均可在对应格式中精确表示。模式为 round-to-nearest, ties-to-even；每个普通加法/乘法都舍入到表中格式，FMA 行的候选只舍入一次。所有非零中间值都位于有限 normal 范围，零单独处理；这里不测试 flush、NaN 或原子调度。

| 规则 | 运算格式 | 输入 \((a,b,c)\) | 参考结果 | 候选结果 | 可以得出的结论 |
|---|---|---|---|---|---|
| ADD-ASSOC | bf16 | \((256,1,-256)\) | 0 | 1 | 不存在该模式下的无条件严格结合律 |
| ADD-ASSOC | fp32 | \((16777216,1,-16777216)\) | 0 | 1 | 同上 |
| MUL-DISTRIB | bf16 | \((3,256,1)\) | 768 | 772 | 不能作为无条件严格分配律 |
| MUL-DISTRIB | fp32 | \((3,16777216,1)\) | 50331648 | 50331652 | 同上 |
| FMA-CONTRACT | bf16 | \((129/128,127/128,-1)\) | 0 | \(-1/16384\) | 分离乘加与 FMA 可能不同 |
| FMA-CONTRACT | fp32 | \((8193/8192,8191/8192,-1)\) | 0 | \(-1/67108864\) | 同上 |

计算方法：使用 Python Fraction 保存精确有理数，对每个普通操作结果按格式舍入；bf16 的有效精度 \(p=8\)，fp32 为 \(p=24\)。对非零 normal 数，令 \(e=\lfloor\log_2|x|\rfloor\)，步长 \(h=2^{e-p+1}\)，将 \(|x|/h\) 舍入为最近整数、平局取偶数，再乘回 \(h\) 并恢复符号。FMA 在精确乘加后应用一次该过程。计算同时检查了输入可表示性及使用的 normal 范围。

这些结果最初由 Python Fraction 计算，现在也已由 [Counterexamples.lean](../VeriTile/Triton/Float/Counterexamples.lean) 中的六条定理复现：`bf16_add_assoc_witness`、`fp32_add_assoc_witness`、`bf16_distrib_witness`、`fp32_distrib_witness`、`bf16_fma_witness`、`fp32_fma_witness`（命名空间 `VeriTile.Triton.FP`）。证明使用 `decide +kernel`，检查的是具体软件 profile 下两侧各自的精确数值输出；没有引入外部数值计算公理。这不是 GPU 符合性实验。

**六条固定输入见证不属于随机采样的 two-gates 结果。** 标量实现的随机差分回归仅用于检查执行语义，不作为 gate 实验。

## 4. 按配置生成接受结果

正式结果表应保存下列字段，空结果使用 null/NOT_RUN，不填零：

机器可读目录位于 [rules.json](../experiments/floating_point/rules.json)。登记工具初始化的记录为 NOT_EVALUATED。[check_numerics.py](../scripts/check_numerics.py) 提供 Python 配置、14 条局部 Triton 实现对、GPU 配对采样和结果导入；导入端核对完整身份、PTX 与统计文件摘要并重新运行两门，生成逐配置数值准入表。哈希与统计重放不认证远端执行真实性，也不关闭 Lean 的外部验证义务。命令、精度支持矩阵和具体变换范围见 [实验目录说明](../experiments/floating_point/README.md)。

| 字段组 | 内容 |
|---|---|
| 规则与适用域 | rule ID、方向、数学前提、参考/候选图、证明引用 |
| 执行实例 | shape/stride、各步 dtype、累加计划、函数实现、编译/启动/设备身份 |
| 输入协议 | probe ID、每个角色的均值/尺度/联合关系、权重、量化/特殊值处理 |
| Bias 结果 | replicate 数、跨 replicate 均值的 local-ULP 均差/标准差/标准误、诊断 z、B、tau、失败或区间不确定的桶、门判决 |
| Vars 结果 | 逐 replicate 最大绝对 oracle 误差和无量纲 K=Ec/Er、return level/U、拟合诊断、门判决 |
| 审计证据 | checker/protocol 版本、原始结果位置、停止原因、时间与最终决定 |

原始主实验由 [report_numerics.py](../scripts/report_numerics.py) 写入运行目录的
`summary.md`、`summary.csv` 和 `summary.json`，覆盖当时的 14 条原子规则 × 3 种精度，
逐行报告最大绝对 z、U、U 的估计类型、两门状态和是否接受。缺失的统计量留空，
接受决定须经过 CPU 回放。每次刷新覆盖当前表，不追加历史表。

每个 probe 配置实际生成独立记录，不能把不同均值或尺度的样本混起来求偏差。规则汇总只覆盖明确列出的配置；未检查的 dtype、shape、计算树、探针或后端均保持未验证。

该表构成原子假设登记与证据查询接口。使用规则时须匹配配置与上下文前提；最终实现等价性由 Lean 在这些假设下推导。整 kernel 的数值复查可以另做，但不充当形式等价性的定义。
