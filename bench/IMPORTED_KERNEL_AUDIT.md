# bug_detection 与 pretrain_kernels 检查记录

检查日期：2026-10-29。逐题记录及源码快照见
[imported_kernel_review.json](imported_kernel_review.json)，用
`python3 bench/check_imported_kernel_review.py` 检查记录是否仍对应当前版本。
该检查不自动判断数学正确性，也不授予数值规则。

## 结论

两组共有 **334 份独立、未解的 Lean 题目草稿**。文件组织、导入独立性和
声明编译可以检查通过；它们尚不能算作 334 个已完成的实际 kernel 等价结果。
每题的 Real 正确性、FP 等价和实际后端连接仍是未解命题。

| 项目 | bug_detection | pretrain_kernels |
| --- | ---: | ---: |
| 题目 | 65 | 269（261 LOCAL + 8 TE） |
| Lean 模块，含两个集合入口 | 457 | 1,885 |
| 独立目录，无跨题 Lean 导入 | 65 | 269 |
| 已证明的主等价结论 | 0 | 0 |
| 已完成的实际源码/后端连接 | 0 | 0 |
| 本集合新增数值规则准入 | 0 | 0 |

`bug_detection` 的分类保持为 30 个负例、20 个代数改写、15 个参数变体。
固定尺寸是各题当前的完整比较范围，不是任意尺寸定理。

## 本轮检查范围

- 检查全部 334 题的七个模块、元数据、集合入口、依赖方向、占位证明、
  literal Triton 程序、具名逻辑输入、输出维度、公开前提及 FP 比较对象。
- 核对 `Correct` 展示的公式与 `Contract.output`，检查 Real IO 的完整坐标
  双射及调度接口。执行、安全、输出和 frame 是待证明判断的组成部分；
  定义这些接口本身不证明程序满足它们。
- 检查 `bug_detection` 的 normalization、RoPE、attention、linear/activation、
  loss/softmax、MoE 六类模板及全部实例配置，包括有限 mask 常数、GQA 索引、
  标签范围、路由 tie policy、capacity 和概率放置。
- 重新检查 `pretrain_kernels` 全部 269 题的两条算术路径，共 **538 个图**，
  记录活跃运算、整数精度和转换。图检查未执行原生 GPU kernel，也不是翻译证明。
- 检查两个 FP 模型、`Allowed`、`D`、`R` 的公开位置，以及后端、有限 Real
  投影和 metadata decoder 尚需完成的连接。没有把导入的数值库算作规则使用。

## 确认并修复的问题

### 1. 19 题的浮点排序缺少 NaN 次序，排序取值还经过求和

旧 rank 比较只处理 `>`/`<` 和 `==`。例如降序输入 `[NaN, 3, 2]`
会得到 rank `[0, 0, 1]`，再产生索引 `[1, 2, 0]`；要求的次序是
`[0, 1, 2]`。它不是仅靠稳定的相等元素比较就能处理的情况。

已对 19 题的实际 literal 程序补齐 NaN 次序和索引 tie break，并把排序值
改为通过选中的索引直接 load。此前用零填充后求和的取值方式还可能改变
负零符号。这项修改没有新增输入限制，Real 数学目标、IO 形状和缓冲区范围保持原定义。

测试覆盖两个排序方向、两个 NaN、正负无穷、重复有限值和正负零构成的小字母表，
长度 1 至 4，共 **9,360 组**。测试检查索引顺序和取值载荷；它不证明完整 kernel、
任意 FP 模型或实际硬件后端的等价。全部受影响题名和位置在逐题记录中。

### 2. 两题的 NaN scratch 存储标签不匹配

`local_triton_topk` 和 `local_triton_topk_dx` 先生成 FP32 NaN，直接写入
声明为 BF16 的 scratch，随后按 BF16 读取。结构解释器按实际存储标签读取；
读错标签会返回默认载荷，不能当作隐式转换。

已在两个 FP 程序及各自的有限 Real 特化中显式转换到 BF16 后存储。
另用 Lean 检查了错误标签读取返回默认值、匹配标签保留载荷的性质。
两题都包含在上述 19 题中，不能加起来统计成 21 道不同题。

### 3. 公式连接与检查器

为 269 题补充 `correctnessProblem` 与规范 `Contract.output` 的可选 Lean
一致性检查；这只连接两份公式，不证明 kernel 正确。

检查器修复了 Python bytecode 进入源码证据、编译期间源码变化后仍写入缓存戳、
基础库仅排除其中一个题集、未指定编译器时仍接受 `--record` 等问题。
同时补齐 `Proofs` 不能反向导入读者入口的检查，并明确区分“记录匹配”和“重新编译”。
四组回归测试分别覆盖两个集合。

### 4. FP 等价步骤的模型证书不能只作为蕴含前提

`GuardedProgram` 的数值步骤现在要求两端实际提供 `Models R` 证书及完整 IO
等价证明。原先的蕴含式允许选择不满足模型约束的中间 backend，使前后两步
都因假前提成立，再通过传递性推出任意程序等价。

`bench/tests/FPGuardedProgramIO.lean` 检查所有推导的执行不变量，包括任意
中间程序、对称和传递步骤；并证明成功的空调度不能与读取未绑定寄存器的
失败调度等价。回归同时检查缺少模型证书会被拒绝、有证书和执行证明的
数值组合仍可使用。该测试已加入 `FPFrontendRegressions`。
这项修复没有解决任何题目的主目标，也没有准入新数值规则。

## 仍需完成的工作

| 范围 | 状态与影响 |
| --- | --- |
| 全部 334 题 | 主目标仍为 `def … : Prop`；没有 `specification` 主证明，也没有已完成的形式反例。 |
| 全部 334 题 | `Proofs/Backend` 接受外部 `native` 参数，尚未绑定并证明固定的实际源码调度。算术 lowering 不能替代实际 tiling、mask、layout、精度、FMA 和归约顺序的连接。 |
| 16 题 | 活跃 int32 数据与 int64 合并到同一整数表示，literal 程序使用 `tl.int64`；需要保留位宽/转换，或给出范围和完整语义连接。 |
| 其中 6 题 | int32/int64 转换被合并后的 carrier 省略，需单独保留转换证据。 |
| 11 题 | Real 入口使用有限特化；它与保留 NaN 的 FP 原程序之间的额外连接仍未解。 |
| 3 题 | 容量条件已公开，FP decoder 与实际数值模型的连接仍未解。 |

整数位宽问题不意味着已构造了这 16 题的输出反例；某些有界 metadata 的转换可以
通过范围证明连接。当前缺少该证据，不能只改 dtype 名字或凭有限样例通过就关闭问题。

16 题为：

`local_fused_permute`、`local_fused_permute_dprobs`、`local_fused_permute_padded`、
`local_fused_permute_with_probs`、`local_fused_score_for_aux_loss_routing_map`、
`local_fused_unpermute_dmerging_probs`、`local_fused_unpermute_padded`、
`local_indices_to_multihot_dprobs`、`local_indices_to_multihot_map`、
`local_indices_to_multihot_probs`、`local_masked_moe_aux_dx`、
`local_masked_moe_global_aux_dx`、`local_masked_moe_seq_aux_dx`、
`local_mtp_roll_packed_fwd`、`local_mtp_tail_mask_closed_fwd`、`local_pad_routing_map`。

另有 66 题的两份算术程序在 scratch 重命名后相同，其中 bug_detection 5 题、
pretrain_kernels 61 题。这不自动构成自比较错误，也不证明两个实际后端等价；
仍须完成各自的源码绑定。

## 验证及证据边界

两组的完整构建范围共 **2,342 个模块**。FP 接口修复后，使用更新的基础库
重新独立编译每题全部模块，并检查两个集合各自的联合导入。
两个集合的全部入口同时导入的额外集成检查也已通过。
各集合的 `validation.json` 记录最终编译器、模块数和源码哈希。

```sh
python3 bench/bug_detection/check.py --verify-record
python3 bench/pretrain_kernels/check.py --verify-record
python3 bench/check_imported_kernel_review.py
```

集合编译记录覆盖题目目录与两个共享适配模块，没有覆盖整个 Mathlib/VeriTile
基础库的源码和编译产物闭包。整体集成后再次独立编译全部 334 题，并通过两个
集合与 FP 回归的联合导入；`main_integration` 另外记录并校验本次共享 Lean
依赖源码、工具链和包清单的哈希，不包含外部包完整源码或编译产物的证明。
完整库构建通过 3,735 个任务，独立比较器接受接口的 6 个目标及回归的 57 个目标。
全仓库 CI 仍有既有的源码审计样例、历史数值绑定哈希及网站依赖审计失败；
这些失败没有被计为通过，也没有修改历史证据来消除哈希不匹配。
本轮使用固定 Lean 4.29.0 和既有基础缓存。
源码快照检查、声明编译和小规模反例测试均不构成实际源码翻译、数学正确性、
数值规则准入或硬件一致性的证明。
