# 公开入口整理与联合检查（2026-10-21）

本轮修正已确认的主／辅助入口混杂，并检查全部 183 题的选定入口是否还有同类遗漏。范围是文件组织、声明保持与当前入口构建；已有数学证明、源码精度、输入契约和数值规则保持。最终检查记录见 [audit.json](audit.json)。

183 题的 366 个 `Correct`／`FPEquiv`（含已选 `FPEquivBound`）路径保持不变。26 题需要组织调整，337 个主入口文件逐字未变，29 个主入口文件作了拆分。58 个迁移声明的完整名称、类型及证明／定义值均保持：其中 20 条辅助结论从 `specification` 改为普通 `theorem`，17 条原辅助定理和 21 个相关定义迁至支持文件。没有删除原结论或缩小证明范围。

当前主入口不再混放普通辅助定理。真实的 forward/backward、producer/consumer、源码分支和多阶段结果保留：34 题仍有多个主结论，逐题理由见 [分类记录](../../../../.veritile-cache/public-entry-alignment-20261021/audit14-multi-headline-reasons.json)。这不是“一题只能有一条 specification”的机械限制。

具体可从以下文件检查：

- `softmax_triton1` 的 [FPEquiv.lean](../../../repairs/tritonbench_g/softmax_triton1/FPEquiv.lean) 仅保留倒数改写的主等价结论；native 结果移至 [NativeObservation.lean](../../../repairs/tritonbench_g/softmax_triton1/NativeObservation.lean)，候选 Real 正确性移至 [ReciprocalCorrect.lean](../../../repairs/tritonbench_g/softmax_triton1/ReciprocalCorrect.lean)。
- `add_example` 的 [FPEquiv.lean](../../../repairs/tritonbench_g/add_example/FPEquiv.lean) 仅保留主等价结论；源码绑定辅助定理位于 [SourceBinding.lean](../../../repairs/tritonbench_g/add_example/SourceBinding.lean)。
- `ksoftmax_triton` 的 native 和空 grid 结果移至 [Native.lean](../../../repairs/tritonbench_g/ksoftmax_triton_grid_y_checked/Native.lean)，原空 grid 适用域完整保留。

任务 README、使用方 imports 和唯一台账的主／辅助声明定位随之更新。辅助模块仍遵守依赖方向：选定 reader 的本地导入闭包没有 `Proofs` 反向导入选定 reader；需要应用公开定理的辅助文件放在任务顶层。详细检查见 [独立复核](../../../../.veritile-cache/public-entry-alignment-20261021/audit14-scope-report.md)。

549 份选定 Python、`Kernels.lean` 和输入契约文件的哈希保持不变。四组修改的 110 份定向构建收据已核对源码、对象和直接依赖。最终联合检查逐题分别导入主入口，以及主入口加辅助模块，并检查保留的公开声明；运行前后对冻结快照作比对。选定根使用当前构建收据或适用的已验收源码／对象 aggregate pins；两个已接受的 minimum 隔离版本保留各自依赖。

首次联合检查的主入口为 183/183，通过辅助联合导入的为 182/183。唯一失败是 BMM 旧、新编译代际对 `Algebra.withDefaultPrecision.eq_1` 生成了不兼容的展开声明；[首次结果和日志](../../../../.veritile-cache/public-entry-alignment-20261021/attempt-01/gate-result.json) 完整保留。随后仅刷新了 backward TensorBridge 的 25 个依赖模块，源码和共享模型未改，失败用例复查通过，见 [定向修复记录](../../../../.veritile-cache/public-entry-alignment-20261021/agent13-bmm-generation-refresh.json)。[修复前后比较](../../../../.veritile-cache/public-entry-alignment-20261021/integration-generation-comparison.json) 确认全部 9,348 个源码节点哈希不变，编译对象变动限定于这 25 个模块。

最终检查从新冻结快照重新运行全部检查，不复用首轮通过结果：**主入口 183/183、主入口加辅助模块 183/183 全部通过**。27,277 个快照路径在检查前后均无漂移，选定根的一致性和结构问题均为零；完整记录见 [最终交付](../../../../.veritile-cache/public-entry-alignment-20261021/public-entry-gate-delivery.json)。检查结束后仅回填台账及报告；回填后的选定路径、主声明和辅助入口与已检查快照逐项对应。

内部依赖复用既有编译基线，没有重放整个历史证明库。初始盘点中的历史直接导入哈希差异保留在诊断记录中，没有把它们改写为全闭包已重新构建的声明。此次联合检查也不等于标准全仓 `lake build`、全体独立源码回放或 GPU 实验。

[唯一任务台账](../../../tritonbench_g/task_progress.md) 的 kernel 验收仍为 **181 verified + 2 verified_with_deferred_numerics**。本轮没有改动数值规则、原数值延期、历史验收、独立回放或 GPU 状态，也没有新增编译器／Torch／运行时证明任务。原语义审查保留；本轮未重新逐题审计全部数学证明。
