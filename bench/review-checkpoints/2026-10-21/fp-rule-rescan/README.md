> 后续结果与当前规则视图见 [共享严格 FP 规则审计](../strict-fp-laws/README.md)。本页及其 audit/validation 保留当时的历史结论。

# FP 数值规则重扫与共享引理（2026-10-21）

**按“严格证明／实验准入”扫描当前 183 题，未发现新的主数值规则证据缺口。** 173 题使用严格模型事实或结构证明，10 题使用已有准入规则；0 题待判定，本轮不新增实验。

此前 43 项“必须把具体模型改成任意 opaque 参数”的待办已撤回。具体模型、严格局部定理和明确的实现/profile 绑定可以直接用于主结果；抽象接口是复用手段，不是新的验收门槛。

## 已完成的共享提取

新增共享模块 [ScalarOpsLaws.lean](../../../../VeriTile/Triton/Float/ScalarOpsLaws.lean)。严格证明集中在这里，原任务保留兼容名称或改为直接引用；没有改变 kernel、Python、输入契约或实验 R。

| 共享定理 | 原消费位置 | 保留的范围 |
| --- | --- | --- |
| [`mul_comm`](../../../../VeriTile/Triton/Float/ScalarOpsLaws.lean#L12) | `int_scaled_matmul` | 所有 Format、FP.Config 和位值；特殊值与NaN规范化均保留。 |
| [`add_positive_zero_fp32`](../../../../VeriTile/Triton/Float/ScalarOpsLaws.lean#L21) | `spinning_lock_reduction` | FP32 ieeeValue；仅两个正零相加。 |
| [`positive_zero_value_fp32`](../../../../VeriTile/Triton/Float/ScalarOpsLaws.lean#L25) | `spinning_lock_reduction` | FP32 正零解码为 some 0。 |
| [`abs_positive_zero_fp32`](../../../../VeriTile/Triton/Float/ScalarOpsLaws.lean#L28) | `int8_matmul_quantization` | FP32 正零取绝对值。 |
| [`add_one_comm_fp32`](../../../../VeriTile/Triton/Float/ScalarOpsLaws.lean#L32) | `fused_activation` | FP32 ieeeValue，任意位值与固定 +1 交换；辅助模型，未改当前主R。 |

乘法交换律用原来的同一证明按格式参数化；没有缩小为有限值域，也没有增加 kernel dtype 版本。正零定理不泛化成任意 `x+0=x`。与 +1 的交换仍是辅助具体模型事实，当前 `fused_activation` 主结果继续使用原来的 ADD-COMMUTE 准入。

## 实验准入规则

| 当前题目 | 实际使用规则 |
| --- | --- |
| `add_example` | FP32 ADD-COMMUTE |
| `add_value` | FP32 ADD-COMMUTE |
| `apply_penalty` | FP32 DIV-MUL-RCP |
| `chunk_cumsum_kernel` | ADD-COMMUTE |
| `chunk_cumsum_vector` | ADD-COMMUTE |
| `fused_activation` | FP32 ADD-COMMUTE |
| `softmax_triton1` | FP32 DIV-MUL-RCP |
| `triton_mul2` | FP32 MUL-COMMUTE |
| `vector_addition` | FP32 ADD-COMMUTE |
| `vector_addition_custom` | FP32 ADD-COMMUTE |

以上复用既有报告与适用条件，未重新实验，也未自动升级为无条件 IEEE 等式。`apply_penalty` 的补充 native FusionRule、`softmax_triton1` 的历史覆盖备注仍按原范围另行记录，不算本次主规则扫描发现的新缺陷。

## 撤回旧标记的依据

- 原 43 项仅据具体模型依赖提出强制接口迁移；这不符合用户最终确认的验收口径，已从现行逐题清单移除。
- `int8_matmul_kernel`：有独立量化整数求和公式；必要的 RTZ 转换、打包权重和 low32 结果没有因接口整理而丢失。转换绑定前提本身不代表数学规格缺失。
- `int_scaled_matmul`：独立 whole-K signed32 公式及明确的 FP32 缩放／INT32 转换后缀匹配当前 API；原结果没有冒称无界实数矩阵乘积。补另一种理想实数结果不是本次规则验收前置条件。
- `fp4_to_bf16`：继续保留独立 E2M1 公式及全表精确转换证明。

上述范围说明不把整数／具体模型定理改称新的理想 Real 定理；只有实际声称的 Real 运算才按 Mathlib 数学语义验收。

## 记录与验证

- [逐题清单](../../../tritonbench_g/task_progress.md) / [JSON](../../../tritonbench_g/task_progress.json) 已更新，每题保留规则证据分类和共享提取记录。既有验收、延期、原始源码缺陷、GPU 与独立回放状态均保留。
- [audit.json](audit.json) 保存全部 183 题的主声明、精确规则／已准入规则、证据引用与共享候选。其余共享候选包括成组的 PTX/格式转换模型；候选不等于证明缺口，不强制把任务专属公式和索引搬入公共库。
- 本轮运行针对性 Lean 构建和共享定理公理检查；检查范围与结果见 [validation.json](validation.json)。没有新增 GPU 实验或独立全库证明回放。
