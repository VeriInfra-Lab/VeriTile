# fp-example 验收报告：浮点模型缺口纠正（2026-10-14）

**41项仅有数学模型结论，不能算完成浮点验证；另外23项只在cast/store处舍入，同样没有补齐算术浮点语义。** 用户接受的是输入、配置和阶段范围，未接受用实数算术替代浮点算术。此前把129项解释成“无需额外数值实验支持”的口径撤回。

本轮修复了分类、验收门禁和逐项缺口清单。**没有新增上述任务的浮点证明，没有通过改标签、原样候选或自反等价把它们算作完成。**

## 当前实际证明层级

| 类型 | 任务数 |
|---|---:|
| 仅数学模型契约：缺浮点/固定宽语义证明 | 41 |
| 仅cast/store舍入模型：缺算术浮点证明 | 23 |
| 存在数值反例：需修订模型或适用域 | 3 |
| 有后端参数化的程序相对等价证明 | 65 |
| 有依赖已有报告前提的程序相对等价证明 | 43 |
| 有条件相对等价证明：精确数值绑定待补 | 3 |
| 缺源码 | 6 |

这些是证明种类和未完成事项，均不是完整GPU正确性通过数。65项有后端参数化的候选/原程序相对等价；43项的相对等价依赖已有实验报告对应的显式规则前提；3项的精确绑定仍待补。两侧可以共享数值误差，相对等价不自动证明原程序满足独立GPU数值规格。

[184项明细](acceptance-tasks.md)、[全部声明模型与数值前提](../../tritonbench_g/numerical_obligations.md)、[具体未补齐的证明](../../tritonbench_g/fp_proof_gaps.md)、[机器可读缺口](../../tritonbench_g/fp_proof_gaps.json)。

## 41项到底缺什么

逐项核对67条公开声明及41项的全部本地Lean文件，未发现可直接公开就补齐的隐藏typed/rounded候选等价证明。互斥分组为浮点算术/归约/递推/旋转21项、数据搬运/比较选择6项、混合整数/浮点6项、量化7项、纯固定宽整数核心1项。

具体缺口逐项标到源码、dtype/cast、所选运算或阶段。纯整数项要补固定宽dot/溢出语义，不能编造浮点实验。未建立准确FP目标或有实质变化的候选前，具体局部数值实验关系记为unknown；不按kernel名称猜测交换律、结合律，也不把待证明的最终输出等式直接当假设。

现有[五个已明确的数值实验配置](../../tritonbench_g/numeric_experiment_handoff.md)继续保留，但只对应已建立条件等价证明的局部绑定，**不是全库剩余工作**。

## 避免按旧标签误分

629条声明按实际模型复核为425条数学契约、33条cast/store舍入契约、167条后端相对等价、4条位解释器契约。旧abstract_rounding中有11条真正的后端结构等价，覆盖8个任务，另有1条BGMV带native加法前提的等价；这些保留其实际证明强度。旧mathematical中isfinite的4条实际使用IEEE位解释器，也单独记录。字节解码、typed命名和局部精确组件不会使整个实数契约自动升级。

31项“数学+abstract且无admitted标签”因此拆为8项后端结构等价与23项cast/store-only。完整来源与49条特殊声明见[模型审计](fp-model-audit/model-level-audit.json)，41项原始逐条审阅见[数学组审计](fp-model-audit/math-only-audit.json)。

## 验收门禁及实际结果

新增 `fp_contract_surface` 必要门禁已实际失败：111项存在公开的浮点后端/位解释器契约，**67个已导入任务没有满足该必要条件，另6项缺源码**。67项包括41数学、23cast/store-only和3全域fp16恒等反例。该门禁即使通过也只说明契约存在，不自动证明完整范围、数值前提或硬件对应。

| 检查 | 结果 |
|---|---|
| 源码、模块归属、公开契约、清单、新模型审阅完整性 | 通过 |
| 全部八库构建 | 6503个Lake任务通过 |
| 工具回归 | 83项通过 |
| FP/固定宽契约必要门禁 | 失败：上述67个已导入模型缺口＋6项缺源 |
| 官方comparator隔离环境及独立重放 | 环境不满足，独立重放未执行 |
| 完整验收 | 未通过 |

[完整收据](acceptance-after-model-review/receipt.json)归档4087个输入，零输入漂移；[归档一致性复核](acceptance-after-model-review-check.json)应为 `integrity_errors=[]`。全部3385个既有Lean输入与此前构建逐字一致。构建通过仅验证当前定理，不能补齐未写出的浮点结论。

`attention_kernel_aligned`、`triton_attention`、`triton_conv2d_fwd` 的全域fp16舍入恒等仍有反例，未改写证明。六个缺源目录仍为fp4_to_bf16、fp4_to_bf16_conversion、layer_norm_triton、multinomial_sampling、spinning_lock_reduction、streamk_matmul。quant_transpose_kernel的已接受片段与完整源码转换阻塞继续分别记录。

[本轮修改与前后哈希](model-review-changes/README.md)、[独立模型/计数及83测试复核](model-review-validation/README.md)、[上一轮报告](ACCEPTANCE.before-model-review.zh-CN.md)均保留。历史收据不改写；旧报告的129口径以本文纠正为准。未提交或推送。

```sh
python3 bench/numerical_obligations.py --check
python3 bench/inventory_tritonbench_g_completion.py --check
python3 bench/acceptance_tritonbench_g.py --check bench/review-checkpoints/2026-10-14/acceptance-after-model-review
```

前两项检查数据/模型审查一致性，应通过；最后一项在归档一致时仍返回非零，因为FP契约必要条件、缺源和独立复验均未完成。
