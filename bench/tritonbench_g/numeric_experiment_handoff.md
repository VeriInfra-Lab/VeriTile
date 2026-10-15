**下面五项只覆盖已有条件等价证明的局部数值前提。** 原41项已为先前接受的配置/阶段补上逐算子或机器数值合同，其硬件对应另见 [41项原语绑定清单](fp_value_native_tasks.json)。仍有23项仅cast/store模型、3项已有全域fp16恒等反例以及6项缺源码，见 [当前证明缺口](fp_proof_gaps.md)。原语验证、数值重写前提和缺失证明分别记录，有限实验不代替普遍性证明。

已有后端参数化等价与已有报告支持的等价都是候选相对原程序的结论；它们不自动证明原程序与独立GPU数值规格一致。

现有条件等价证明已经明确的局部待办是 **5 个数值配置的实验或精确绑定**，涉及 `context_attn_fwd`、`context_attn_llama` 和 `bgmv_expand_slice`。两项 context 共用一个 typed-fp32 乘法配置；BGMV 有四个不同的加法配置。尺寸、布局、访问安全、有限输入及声明中限定的配置/阶段按已接受范围保留，不作为新增数值实验待办。

另外有三条全域 fp16 舍入恒等前提需要修订模型或适用域；它们不能靠增加抽样“验证通过”。共享的 `RoundingModel` 结构要求单列一项。以下计数按数值配置去重，不按任务或声明重复计数。

**待补的 5 个配置**

记 `E = FP.ProfiledRewrite.engine M`。`E.binary p d op a b` 表示精度标记为 `p`、数据通道为 `d` 的实际标量操作；`none` 经引擎采用默认 fp32 精度，但不会把 `.fp16`、`.bf16`、`.fp32` 通道变为 `.real`。实验要核实对应源码实际编译出的操作，不能只按“fp32”名称合并。

| 配置 | 精确待验证关系，均要求 `finite(a) ∧ finite(b)` | 适用位置与待办 |
|---|---|---|
| typed-fp32 MUL-COMMUTE | `M.binary (some .fp32) .fp32 .mul a b = M.binary (some .fp32) .fp32 .mul b a` | 两个 context 的 `scaleFirst_equiv`；验证 `TypedScalarArithmetic.entry` 的独立 `:typed-fp32` 绑定，交付 `validatedTyped` 所需证据。 |
| BGMV ADD，显式 fp32 求值精度 / fp16 通道 | `E.binary (some .fp32) .fp16 .add a b = E.binary (some .fp32) .fp16 .add b a` | 72 种源码配置中的 28 种；需要该实际操作的 probe 与 native 绑定。 |
| BGMV ADD，显式 fp32 求值精度 / bf16 通道 | `E.binary (some .fp32) .bf16 .add a b = E.binary (some .fp32) .bf16 .add b a` | 4 种；既有 bf16 输入先扩成 fp32 再显式舍入的实验，不能直接代替这个 native 操作。 |
| BGMV ADD，显式 fp32 求值精度 / typed-fp32 通道 | `E.binary (some .fp32) .fp32 .add a b = E.binary (some .fp32) .fp32 .add b a` | 8 种；需要独立 typed-fp32 加法绑定，旧 `.real` 通道规则不足。 |
| BGMV ADD，显式 fp64 / real 通道 | `E.binary (some .fp64) .real .add a b = E.binary (some .fp64) .real .add b a` | 12 种；需要实际 fp64 操作的 probe 与绑定，不能用 fp32 报告替代。 |

上述 `some .fp32` 是求值精度上下文；`.fp16`、`.bf16` 和 typed `.fp32` 仍分别保留其实际运算通道，不能改用 fp32 加法后再次舍入来冒充。BGMV 的配置数来自完整 Lean 源码的 input/weight/cast/output 组合抽取，不表示这些组合已全部完成目标 GPU 原生编译。实验先确认实际配置可编译，并保留不支持配置的记录。

context 的 `a` 是原始 dot 得到的 typed-fp32 `qk`，`b` 是实际 typed-fp32 scale；比较点在后续 real 通道转换之前。fp16/bf16 存储路径必须保留原始输入转换、dot 输出类型和 scale 类型。源码中的 `where` 会先求值其操作数，掩码外仍被求值的乘法 lane 也必须检查有限性。旧 fp32 数值报告和已有离线编译可作为来源与类型证据；离线编译本身不补齐 typed 绑定，旧报告 Triton 3.7.1 与已有编译证据 Triton 3.8.0 也应分别记录。

BGMV 的 `a` 是实际乘积归约结果，`b` 是已加载的旧输出；只在非 sentinel、实际到达的动态循环及 `ADD_INPUTS` 分支请求交换律。必须保持原来的 cast、归约、load mask、输出转换，并检查所有实际求值 lane，包括最终 store 掩码不写出的 lane。不能仅采样原始输入张量来替代这些中间操作数。无需把整个 dot、归约或 BGMV 新增为一个准入原子。

精确来源：[TypedScalarArithmetic.lean](../../VeriTile/Triton/Float/TypedScalarArithmetic.lean)、[BGMV 的 ScalarAdditionCommutes / Site.Holds / NativeAdditionCommutes](bgmv_expand_slice/Proofs/NativeFP.lean)、[BGMV 现有实验格式绑定](bgmv_expand_slice/Proofs/AdditionProfiles.lean)。

**已有结果，可在原配置与显式模型前提下复用**

下表的 `Q32` 简写保留 fp32 运算/舍入位置；精确实现与绑定以报告和 Lean 片段为准。

| 规则 | 表达式与域 | 现有证据 |
|---|---|---|
| ADD-COMMUTE | `Q32(a+b)` 对 `Q32(b+a)`；`a,b` 有限 | [主报告](../../experiments/floating_point/report/summary.json) 的 fp32 行通过。 |
| MUL-COMMUTE | `Q32(a*b)` 对 `Q32(b*a)`；`a,b` 有限 | 同一主报告的 fp32 行通过；原绑定是 `.real` 通道，不自动覆盖 typed-fp32。 |
| DIV-MUL-RCP | `Q32(a/b)` 对 `Q32(a*Q32(1/b))`；`a,b` 有限且 `b ≠ 0` | [补充报告](../../experiments/floating_point/supplement/report/summary.json) 的 fp32 行通过；这里是普通 Triton `/`，不是 `tl.div_rn` 的 DIV-RCP。 |

BGMV 当前有 **1 个可条件复用的 profile**：`(some .fp32, .real)` 共 20 种配置。前端修复保留了窄输出转换内的 fp32 作用域，原先 8 种 `none/.real` 与 12 种显式 fp32/real 因而合并；引擎默认精度与显式 fp32 的加法定律定义相等，未增加或免除数值条件。`NativeFP.fp32_scalar_of_rules` 与 `fp32_native_of_rules` 给出到现有加法规则的桥接；实际操作数有限及报告/后端模型对应前提仍须满足。

已有报告的完整性与指定行已复核，本轮没有重新执行 GPU 数值实验。PASS 适用于记录中的实现、输入分布、精度、编译器和设备；迁移到新环境需复跑对应配置并重新绑定。有限输入不保证结果有限，不能过滤溢出结果后再宣称通过。

**三条全域 fp16 恒等前提：修订模型，不新增“测通”任务**

以下声明均要求 `R.round .fp16 = id`，即 `∀ x : ℝ, round_fp16(x) = x`：

- [attention_kernel_aligned_io_correctness](attention_kernel_aligned/AttentionKernelAligned.lean)。
- [triton_attention_forward_io_correctness](triton_attention/TritonAttention.lean)。
- [triton_conv2d_fwd_io_correctness](triton_conv2d_fwd/TritonConv2dFwd.lean)。

普通 binary16 舍入下，精确 fp32 输入 `x = 1 + 2^-11 = 1.00048828125` 按 ties-to-even 舍入为 `1`，已否定该全域恒等前提。这不否定条件 Lean 定理；它说明真实舍入不能满足这个前提。只测试已量化 fp16 再转 fp16，检查的是幂等性。

处理方式是保留每一步舍入、证明相应参考公式，或将结论明确限制到所有实际 cast 操作数均可精确表示的域，并提供这些中间值的证据。输入本身是 fp16，并不保证乘积或 softmax 概率仍可精确表示。若改用误差容忍的实验接受契约，也需要相应修改结论，不能据此撤掉原精确等式的前提。

**共享模型要求：1 项**

33 条公开声明使用 `RoundingModel`，其结构内含：

```text
round_real : round .real = id
round_idem : ∀ dtype x, round dtype (round dtype x) = round dtype x
```

这是同一个模型的两项字段，不是 33 个独立实验缺口。`.real` 恒等指数学通道；具体舍入实现需要为幂等性提供对应证据，并明确有限域、溢出、subnormal/flush-to-zero 等处理。可检查各格式有限可表示值的重复转换及边界值，但有限抽样不等于证明对所有实数成立。模型本身只对 cast/store 插入舍入，其他数学算子与真实硬件的对应关系仍单列为语义边界。

**执行与交回结果**

先为上述 5 个配置定位实际源码操作并确认输入/计算/输出通道、cast 和 intrinsic，再选择复用报告与新增 probe。现有框架可执行已有三条标量规则；**目前不能用下面的旧规则命令自动完成五个新配置的 probe 或 typed/native 绑定**。扩展 probe 时必须给新实现/绑定独立身份，并使 importer 能校验该身份。

在准备好的 NVIDIA GPU 环境，从仓库根目录运行现有规则复验：

```bash
python scripts/check_numerics.py check
python scripts/check_numerics.py run --rules ADD-COMMUTE,MUL-COMMUTE --formats fp32 --output Logs/assumptions-main
python scripts/check_numerics.py import Logs/assumptions-main --output Logs/assumptions-main-import.json

python scripts/check_numerics_supplement.py check
python scripts/check_numerics_supplement.py run --rules DIV-MUL-RCP --formats fp32 --output Logs/assumptions-supplement
python scripts/check_numerics_supplement.py import Logs/assumptions-supplement --output Logs/assumptions-supplement-import.json
```

原采样分布之外，新增 probe 应覆盖实际 kernel 操作数分布、正负零、subnormal、最小 normal、最大有限值、舍入中点邻域及有限输入产生溢出的情况。有限性/非零域判定应与声明一致；域外样本单独统计，域内产生的非有限结果保留。位级差异与 ULP/误差统计分开报告，不用统计 PASS 覆盖精确反例。

每个配置交回完整结果目录及 receipt，至少包含：

1. obligation/profile ID、受影响声明、精确左右表达式、finite/nonzero guards、实际改写位置与 eager lane 范围；input/compute/accumulator/output dtype、Lean 通道、默认精度、cast/intrinsic。
2. 原始源码与 probe 哈希、各侧 TTIR/PTX/二进制哈希、原源码到 probe 的操作对应；GPU 实际型号/capability/driver、Triton/CUDA 版本、launch 参数、融合和数学选项。编译不支持和运行错误保留。
3. 输入分布、种子、量化方式、边界样本与实际操作数样本；valid/skipped 数量、原始差异/反例、`record.json`、`observations.npz`、manifest 及文件哈希。
4. 协议与阈值、bias/magnitude 各 gate 的统计和 decision、独立 CPU import/replay 输出、退出码与哈希，以及绑定验证结果。SMOKE_ONLY、失败或未决结果不计准入。

现有 `observations.npz` 支持统计重算，不是原始 GPU 输出逐项重放；精确反例及新 probe 所需的位级比较应另行保存。只交 summary 的 PASS 字段不足以复核。实验能支持明确配置与域下的经验准入或发现反例；全域 IEEE 等式和完整硬件语义证明仍需独立证据。
