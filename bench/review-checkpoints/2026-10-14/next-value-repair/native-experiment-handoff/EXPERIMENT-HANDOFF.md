# 数值实验交接补充（候选，尚未执行 GPU）

**37 个 native 工作项完整对应 V3 的 32 个扩展任务，加 3 个原41项 maximum 迁移和 2 个 retention 参考定义修订。原来 5 个局部数值配置仍独立待绑定，不能被这 37 项替代。** 已接受的正常输入、配置及阶段限制不重新计为“缺数值假设”；记录尺寸、布局和启动参数是为了使实验可复现。

本补充提供精确来源和配置清单，不声称已经有 37 套可直接运行的 GPU 验证器：

- 17 个任务有归档的离线编译配置，共 111 条：91 条编译成功、20 条失败。它们都没有 GPU 数值执行结果。
- 其余 20 个任务尚无本清单已定位的离线编译 profile；先记录真实源码编译结果，不能预设 dtype/config 可编译。
- 37 项都没有在本清单中交付“整个工作项的 GPU 采集器 + 独立批量 expected evaluator”。逐项状态已明确为未就绪。Lean 参数化合同或固定见证不能冒充这样的 evaluator。
- 已有的 fp32 标量探针可以直接复用，见下方命令；它不覆盖所有 dtype、归约树、超越函数、dot 或整核来源绑定。

## LLaMA 当前来源修正

`llama-current-binding.patch` 是可审查的三文件候选补丁，覆盖 generator、JSON 和 Markdown。它只改变 `flash_decode2_llama` 这一项：

- 当前 `model_receipt` 指向 `maximum-flash-decode2-llama/receipt.json`，绑定真实 typed-fp32 源码与 parsed `maximumFloat` 的六条公开阶段规格。
- 旧 `flash-stages/receipt.json` 完整保留为 `previous_model_revision`，不再作为当前 maximum 的 expected 模型。
- 明确原接受的三个 supplied-state 阶段：normalization store、final copy、maximum update；不扩成完整 online loop。原始完整核的 exp/FMA 编译证据只是上下文证据。

`before/` 保存旧清单字节；`after/` 是候选新文件。generator 已用真实包重放，仅将输出重定向到 `/tmp`，验证其余 36 项及全局字段完全不变。没有改 V3 的冻结 native-plan 快照、registry 或任何证明文件。

## 逐项配置矩阵

- `native-experiment-matrix.json`：37 个任务、五个局部规律、完整公开合同/来源信息、输入场景、实际要采集的中间量/输出及未指定字段。
- `COMPILE-PROFILES.tsv`：111 条历史编译记录的 source variant、signature、constexpr、options、target、版本、结果、证据路径/SHA/JSON pointer。
- `TASK-MATRIX.md`：37 项简表。

JSON 中 `missing_compile_fields` 的含义是归档记录没有填写该字段，不能按常见默认值猜测。`options={}` 表示显式 options 为空；**不等于已逐项记录 resolved defaults**。运行时仍需捕获 compiler metadata 的实际 warps、stages、FP32 mode、fusion、FTZ 等配置。

原始 Python 在工作区存在时给出其真实路径和 SHA；六项历史缺源码任务的可用副本按实际情况给出包内文件或 `inputs.zip` 的精确 member 与 SHA。从 zip 取出代码也要再次核对 member SHA；原始代码与编译候选分开。

## 每个实验包应如何判定

1. **固定对象。** 选择一个 task/stage、原始源或明确候选、具体 dtype/config、一个公开合同与一个软件 profile。保存源码/候选 diff、Python/typed source/参考实现 SHA。不能把候选编译成功归给原始代码。
2. **先确认图。** 保存未加探针原核的 IR/PTX/cubin。对照合同的 cast、dot/dotAcc、mul/add/FMA、操作数顺序和 reduction seed/tree。若图不同，使用已有独立的 compiled-graph 合同；没有该合同则记录 `GRAPH_CONTRACT_MISSING`，不能假设融合等价，也不能将其改名成“待实验为真的交换/结合律”。
3. **填写实际配置。** 矩阵中的编译 signature/constexpr 可以复用；运行时 M/N/K、strides、pid/grid、各 allocation 大小、alias、初态 words、实际 GPU/driver/CUDA 版本及 resolved compiler options 仍须显式填写。软件见证的小 tile 不一定是合法 tensor-core launch，必须分别标记。
4. **给出独立 expected。** 对闭合软件 profile，先实现并绑定批量 oracle，输入为初态 words 和固定参数，输出为中间/最终 words；逐例核对已有 Lean 见证，不能读取 native 输出再回填 expected。对 exp/rsqrt/dot/host encoder 等剩余回调，先明确具体函数/指令语义或误差关系及其域；“任意 callback”本身没有可比对的 expected。仅采集这些原语的输入输出可记为 `CHARACTERIZATION_ONLY`，不能记整核相等。
5. **采集实际操作数。** 包括循环内每次到达站点、where 的 eager lanes、最终 store mask 外仍求值的 lane、padding 乘积、常量转换和 supplied-state。不要只采样最初的输入张量。保存原核→插桩核站点表，复核插桩没有改变待测运算图；无法确认时只归类为孤立 primitive probe。
6. **比较位值和帧。** 按真实宽度比较 8/16/32/64-bit words，保留 NaN payload、sign、±0、subnormal、Inf、overflow。不得用 allclose、统一 NaN 或忽略负零来制造 PASS。检查声明中所有输出及未写地址；多输出如 Mean/RSTD、saved activation、DQ/DK/DV 分别检查。
7. **明确未定义策略。** 模型允许任意 byte/undef 的位置只检查已声明的集合/类型/帧条件；不能把一次观察的值升级成恒为零。未知 masked-load 值也不得直接换成零。

允许的最终标签：

| 标签 | 意义 |
|---|---|
| `COMPILE_FAIL_ORIGINAL` | 原配置编译失败；原始失败保留，候选另记。 |
| `COMPILED_ONLY` | 编译成功和图证据，不包含数值执行。 |
| `PROFILE_OR_EVALUATOR_MISSING` | 缺精确 profile、有效图合同或独立 expected adapter，尚不能比较。 |
| `CHARACTERIZATION_ONLY` | 收集 callback 的原语行为，没有独立预定 oracle。 |
| `MISMATCH` | 在已声明域内发现原始位差异；保存完整反例和配置，不过滤掉。 |
| `MATCH_ON_TESTED_INPUTS` | 仅该配置/已记录输入的经验匹配，不是普遍量化定理、整核编译器证明或其他配置准入。 |

SC/原子顺序、race freedom、fairness、termination、allocation safety 与 Python→typed source 对应是独立语义工作。事件采集有用，但标量位测试不能补齐它们；37 条中的 spinning、layer norm、StreamK 尤其必须保留这一边界。

## 原来的五项：实际需要比较什么

设 `E=FP.ProfiledRewrite.engine M`。所有下列域要求仅限原声明已有的实际操作数有限性，不自动过滤溢出结果。

| 配置 | 精确待绑定关系 | 实际站点 |
|---|---|---|
| typed fp32 MUL | `M.binary (some .fp32) .fp32 .mul a b = M.binary (some .fp32) .fp32 .mul b a` | 两个 context 的真实 typed-fp32 dot 输出与 scale；包含 eager lanes。 |
| BGMV fp32/fp16（28 配置） | `E.binary (some .fp32) .fp16 .add a b = E.binary (some .fp32) .fp16 .add b a` | 实际 product-reduction 结果与旧输出。 |
| BGMV fp32/bf16（4 配置） | 同式，dtype 为 `.bf16` | 同上；不得以 widen→fp32→narrow 替换实际运算。 |
| BGMV fp32/fp32（8 配置） | 同式，dtype 为 `.fp32` | 同上；旧 `.real` 规则不能自动提供 typed 绑定。 |
| BGMV fp64/real（12 配置） | 同式，precision 为 `some .fp64`、dtype 为 `.real` | 同上；需原源码明确的真实 fp64 lowering，不能仅凭 `.real` 名称。 |

BGMV 只在非 sentinel、实际到达的动态循环和 `ADD_INPUTS` 分支比较交换顺序，保持 cast/reduction/load/store 图。正常尺寸、配置/阶段范围不是新增待证规律。每个 profile 都需要真实 site probe 与绑定；现有 runner 只是可复用框架，不能宣称它已经覆盖五项新绑定。

`ROUNDING-MODEL-STRUCTURE` 是共享模型对应要求，不重复算作每个数学定理的实验欠账。三个旧全域 `round(fp16)=id` 已由 `1+2^-11 → 1` 反驳，新值合同保留舍入；该全域命题不在待证实验中。

## 可以直接复用的现有命令

下列命令尚未在本次审查中执行。工具文件和参考源码 SHA 另列于 `reusable-tools.json`。

已有 fp32 标量位 oracle 与 GPU probe（add/sub/mul/div/neg/abs；普通 `/` 与 `div_rn` 分开）：

```bash
python3 experiments/floating_point/value_validation/run_scalar.py \
  --generate-reference /tmp/scalar-reference-NEW.json
python3 experiments/floating_point/value_validation/run_scalar.py \
  --run /tmp/scalar-reference-NEW.json --output /tmp/scalar-native-NEW
```

第一条需要已有 Lean 环境；第二条需要 CUDA/PyTorch/Triton 与实际 GPU。它们生成原始位 mismatch，且输出路径必须新建。**这是 scalar32 primitive profile，不是37项整核 harness，也不是五项 site binding 的自动完成。**

已有代数关系统计框架可复跑原 `.real` 通道 fp32 报告配置，保留其统计判定和 CPU replay；仍不能代替新 typed/bf16/fp64 site probe：

```bash
python3 scripts/check_numerics.py run --rules ADD-COMMUTE,MUL-COMMUTE \
  --formats fp32 --output /tmp/scalar-laws-NEW
python3 scripts/check_numerics.py import /tmp/scalar-laws-NEW \
  --output /tmp/scalar-laws-replayed-NEW.json
```

已有 SGMV 原始/逗号修复候选离线编译 probe 可原样复制到新临时目录运行；它只编译、不执行 GPU，也没有 expected evaluator：

```bash
experiment_probe_dir=$(mktemp -d /tmp/sgmv-native-replay.XXXXXX)
cp bench/review-checkpoints/2026-10-14/next-value-repair/sgmv-expand-slice/native-probe/probe.py "$experiment_probe_dir/probe.py"
python3 "$experiment_probe_dir/probe.py"
```

同样可替换为 `triton-attention-value/native-probe/probe.py` 或 `matmul-leakyrelu/native/probe.py` 进行各自已归档配置的离线复编。它们当前硬编码了本工作区原始源码根路径；换机器时需显式修正并记录脚本 SHA，不能说是原样重放。不要在冻结包目录直接运行这些脚本。脚本退出 0 只表示收集完毕，逐配置 `FAIL` 仍是失败；原 SG tuple、FP4 int16 pow、linear fast_gelu、部分 attention dtype、StreamK int8 的已知失败不得被删掉。

## 禁止改成“等待实验验证为真”的已知反例

- 全域 fp16 舍入恒等；half 再 cast half 的幂等实验不是它的证明。
- source 的 dot+add/split mul+add 与 dotAcc/FMA 在全域相等；已有有限输入反例。
- compare/select 与 maxnum 在含 NaN 域相等；右 NaN 反例已知。
- padding 的 `0*Inf` 总可删、缺省 undef 总是零、某次未定义 byte 是普适值。
- 任意 Nat lane 的 int32 cast 等于理想自然数；`2^31` 的 signed-low32 反例已知。
- 某些有限 SC schedule 成功就说明原 GPU 程序 race-free、终止或满足任意 scheduler。

这些情况要保留真实图/位语义、既有候选或明确的独立语义工作，不让用户反复采样直到“通过”。
