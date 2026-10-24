# 共享严格 FP 规则与当前准入视图（2026-10-21）

本记录接续 [FP 规则重扫](../fp-rule-rescan/README.md)，只核对有限规则目录和受影响消费者的差分，不重扫 183 题。当前规则导出已验证：81 条历史记录对应 78 个实例，其中 42 个具有严格模型证书，36 个仍属于实验准入。八个已选消费者的严格证书迁移与最终验证已完成；唯一任务台账已同步规则证据分类。

严格结果限定于明确的数值模型、精度、转换、输入条件和输出观察。使用具体模型不要求额外全硬件一致性证明；它也不意味着所有硬件实现自动满足模型等式。历史实验、原有延期及验收状态保留。

逐实例证据与消费者差分记录在 [audit.json](audit.json)，检查和构建凭据记录在 [validation.json](validation.json)。当前目录证书已通过 live Lean 导出及 104 个项目源码／对象／导入收据的递归核验；34 个外部 Lean/Mathlib 编译依赖作为明确的基线边界保留。

当前分类以 [CurrentAdmission.lean](../../../../VeriTile/Triton/Float/CurrentAdmission.lean) 的 proof-carrying `StrictEntry` 为准。每个条目持有对应源码关系的证明，并匹配输入、运算、累加、输出四种精度、guard、整数界限及 load/rnd/store 路径。42 个严格实例覆盖 44 条历史记录；36 个实验实例对应其余 37 条。历史表中 FP32 的 `EXP-SUB`、`EXP-ZERO`、`EXP-NEG-INF-SUB` 各重复一行，因此历史行数与实例数不同。五份 `*Admission.lean` 和原实验报告保持不变。

旧 `ReportedAdmission` 原始行没有 guard 字段；当前视图按当前源码和策略适配，并逐行记录来源。历史采样分布不替代公开适用域，也不成为全域逐位相等的证明。

| 严格证书族 | 实例数 | 原始目录范围与条件 |
| --- | ---: | --- |
| 标量交换律与乘除单位元 | 12 | ADD-COMMUTE、MUL-COMMUTE、MUL-ONE、DIV-ONE，各三个精度配置；保留原有限域及转换／存储 |
| maximum | 12 | MAX-COMMUTE、MAX-ASSOC、MAX-IDEM、MAX-NEG-INF，各三个配置；实际 probe 的每个 maximum 节点均为 FP32，只在最终 store 转换 |
| 精度恒等式 | 8 | ROUND-IDEM ×3、BF16-WIDEN-RETURN ×2、BF16 输入 CAST-MOVE ×2、FP32 输入 ACC-WIDEN ×1 |
| 计数转换 | 2 | COUNT-ZERO；COUNT-SUCCESSOR 保留 `0 ≤ i < 2^24` 与 int32 不溢出界限 |
| 指数特殊值 | 8 | tl.exp 的 EXP-ZERO／EXP-NEG-INF-SUB 各三个配置；libdevice 对应两条 FP32 规则 |

这些证书使用明确的软件模型。目录的标量和转换配置采用 nearest-even、无输入／结果 flush 的 `ieeeValue`，保留正负零和 subnormal；maximum 使用 maxnum、两种零相遇时选择 +0。普通 Triton `/` 需要所选除法模型绑定，没有替换成 `tl.div_rn`。指数模型分别实现文档规定的 PTX `ex2.approx.f32` 与 libdevice 特殊值分支；tl.exp 前置的 FP32 `log2(e)` 常量乘法仍在程序内，有限非零输入的近似行为保持参数化。详见 [特殊值模型说明](../../../../documents/FloatingPointSpecialValues.md)。

36 个剩余实例的完整精度、guard 和原证据索引见 [当前导出](../../../../.veritile-cache/strict-fp-laws-20261021/current-registry.json)。其中 33 个是一般不成立精确等式的放宽改写，另有三个 ADD-ZERO 实例：原有限域包含 -0，而 nearest-even 下 `(-0) + (+0)` 为 +0，不能据此消除原逐位关系的实验义务。两个 BF16 输入 ACC-WIDEN 也继续保留准入；CAST-MOVE 的严格结果只覆盖原有 BF16 输入配置。

`≡[[]]` 表示没有实验准入规则，仍可使用程序视图固定的严格软件理论；它不表示任意 opaque 运算满足这些恒等式，也不自动给出任意 `Models` 的非空性。局部证书必须接回实际 kernel 的执行、输入域、输出及 frame。FMA、分开乘加、运算顺序和源码精度继续区分。`#print_fp_assumptions` 分开报告严格证书与实验规则。

本轮只改变以下八题的规则证据分类，均对准台账选定的 repair 入口，并保留原程序、精度、输入域、独立 Real 证明、执行／输出／frame 连接：

| 任务 | 替代实验准入的严格规则 |
| --- | --- |
| [add_example](../../../../bench/repairs/tritonbench_g/add_example/FPEquiv.lean) | ADD-COMMUTE |
| [add_value](../../../../bench/repairs/tritonbench_g/add_value/FPEquiv.lean) | ADD-COMMUTE |
| [chunk_cumsum_kernel](../../../../bench/repairs/tritonbench_g/chunk_cumsum_kernel/FPEquiv.lean) | ADD-COMMUTE |
| [chunk_cumsum_vector](../../../../bench/repairs/tritonbench_g/chunk_cumsum_vector/FPEquiv.lean) | ADD-COMMUTE |
| [fused_activation](../../../../bench/repairs/tritonbench_g/fused_activation/FPEquiv.lean) | ADD-COMMUTE |
| [triton_mul2](../../../../bench/repairs/tritonbench_g/triton_mul2/FPEquiv.lean) | MUL-COMMUTE |
| [vector_addition](../../../../bench/repairs/tritonbench_g/vector_addition/FPEquiv.lean) | ADD-COMMUTE |
| [vector_addition_custom](../../../../bench/repairs/tritonbench_g/vector_addition_custom/FPEquiv.lean) | ADD-COMMUTE |

两条 cumsum 核验的是实际 `FPEquiv.memory_equiv`，不是原目录中的辅助 `chunk_sum_left_equiv`。其原 carry 域仅约束会被后续输出读取的操作数，最终未使用 carry 不增加条件。`fused_activation` 的 native 结论显式绑定实际 `add32` 到 `FP.add cfg`，保留 FMA 图、其余原语端口与原有域；完整 IO、安全性、成功执行及 frame 均已接通。示例 `HyperConnectionsWidth.mhc_width_equiv` 也迁移到空实验规则集，其 matrix 结论仍保留 DIV-MUL-RCP。

因此，[唯一任务台账](../../../tritonbench_g/task_progress.json) 的**规则证据分类**由 173 strict/structural + 10 admitted 更新为 **181 exact_or_structural + 2 admitted_numerics**。其余 175 题沿用历史审计分类，只更新报告链接；剩余 `apply_penalty` 与 `softmax_triton1` 保留 FP32 DIV-MUL-RCP、原适用条件及数值证据。**Kernel 验收状态**仍为 181 verified + 2 verified_with_deferred_numerics；这是另一套统计，全部验收、源码精度、历史数值延期、独立回放和 GPU 字段均未修改。

最终受影响依赖检查覆盖 68 个入口；主工作区闭包含 837 个模块，核对 755 份当前工作区模块收据，并明确记录 82 个复用编译基线边界，未发现未解决问题。隔离版本另核验 18 份模块收据及 11 个编译基线边界。已接受隔离版本的入口按其自身版本核验，不能用当前主工作区依赖图替代。六个保留数值放宽的示例另有 18 份源码的非注释 token 不变检查。台账重新生成与 `python bench/task_progress.py --check` 均通过；审计外字段的规范化哈希保持不变。

55 项目标 Python 回归通过：43 项规则／诊断断言，另有 primitive 的 9 项与 matrix 的 3 项。初次失败的两处 rational `abs` 夹具保留失败日志，修正后定向重跑通过；计数的两个 headline 各自保留原 300 秒限时及精确的 7 个 empirical／6 个 strict 名称断言。临时 harness 只替换 Lake 子进程：`lake env lean` 使用工作区 Lean 检查，`lake build` 使用已构建 overlay 的导入检查；依赖实际构建由另行核验的收据证明，不能将该替换称为标准 Lake 初始化或构建成功。见 [Python 回归结果](../../../../.veritile-cache/strict-fp-laws-20261021/expectations08-result.json)。

验证使用专用 Lean 工具链和工作区收据；复用编译对象不记为独立源码回放，也不构成全仓标准 `lake build` 成功声明。完整范围、版本边界和收据见 [最终验证汇总](../../../../.veritile-cache/strict-fp-laws-20261021/validation-summary.json) 与 [validation.json](validation.json)。本轮没有新增数值实验、GPU 实验或全体 183 题的独立证明回放。

规则目录可复现导出如下；命令构建并求值 Lean 当前视图，只写新的导出文件：

```sh
python scripts/export_current_fp_rules.py --output /tmp/current-fp-rules.json
python scripts/fp_rule_inventory.py --current
```

当前工作区可给两条命令补充 `--receipt .veritile-cache/strict-fp-laws-20261021/build/lib/lean/VeriTile/Triton/Float/CurrentAdmission.workspace-build.json --lean /tmp/veritile-lean-toolchain/bin/lean`。不带 `--current` 的 inventory 保守保留历史 81 行／78 实例，不凭候选名称排除任何实验义务。完整检查还拒绝精度、guard、计数界限、证书匹配及源码／对象哈希篡改，见 [导出验证日志](../../../../.veritile-cache/strict-fp-laws-20261021/current-registry-validation.txt)。
