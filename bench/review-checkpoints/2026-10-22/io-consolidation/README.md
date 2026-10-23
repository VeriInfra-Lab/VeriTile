# TritonBench-G IO 组织与去重复核（2026-10-22）

本轮从 `uniform_sampling/Correct.lean` 中重复的 `samplingIO` 出发，检查唯一逐题清单选定的全部 183 题。检查对象是 IO 数据的职责、复用和 reader 组织；整题证明完成状态、输入域、程序、精度和数值规则保持不变。

| 项目 | 结果 |
| --- | ---: |
| 逐题审查 | 183 |
| 有最终组织修改的题 | 117 |
| 最终无需修改的题 | 66 |
| 解析 re-export 后的 reader 文件 | 367 |
| 保留的主 specification | 449 |
| 从 reader 移出或改为共享实例的完整 IO record | 47 |
| 当前发现但未处理的本轮 IO 组织问题 | 0 |

117 题的改动包括 reader 完整记录复用、纯 IO 数据从混合证明文件移出、候选调用描述复用，以及少量连接证明归回 `Proofs/`，并非 117 题都存在相同的重复记录。未把所有历史、非选定 helper 重新整理作为本轮前置任务。

## 结构与示例

同一个具体 IO 接口只维护一份完整描述。题级 `IO.lean` 或明确命名的 `RealIO.lean`、`NativeIO.lean`、`PairIO.lean` 等 IO 模块存放坐标、位置、mask、运行参数和调度数据；reader 可以保留薄实例。执行、安全和公式连接证明留在 `Proofs/`。实际不同的 encoding、channel、observer 或调度保留不同视图，并逐题解释差异。

`uniform_sampling` 的 [IO.lean](../../../repairs/tritonbench_g/uniform_sampling/IO.lean) 现在维护 `KernelOffsets.output`、完整的 `KernelPublic.io` 与候选 `KernelPublic.variant`。[Correct.lean](../../../repairs/tritonbench_g/uniform_sampling/Correct.lean) 只选择 Real 解释：

```lean
abbrev samplingIO := KernelPublic.io realAlgebra
```

原完整泛型 IO、候选、Real IO 和输出位置函数的四个 `rfl` 对照均通过；原 Real / FP 主定理保持原文。`Proofs/PublicIO.lean` 留下数学、安全、执行和等价连接证明。

## 验证与防止回退

[audit.json](audit.json) 逐题记录 canonical IO、实际不同视图的说明、改动文件、源哈希与精确验证证据。[reader-roles.json](reader-roles.json) 保留此前独立 Real／实际程序对的角色判断，记录此次数据整理后的声明位置与哈希；旧角色审查报告保持冻结。另修正旧报告中 `attention_kernel_aligned` 两条短名相同、完整 namespace 元数据误指向 BF16 分支的问题，两个精度分支的四个主结论均保留。

修改题均有旧完整 IO／数据定义的相等检查、针对性构建和 fresh 主入口加受影响辅助入口联合检查。构建复用适用的既有对象，不等于从零重放全部证明。未改题按源码阅读与哈希复核验收；`max_reduction_triton` 曾尝试的非必要 observer 搬迁已撤回，恢复与尝试日志单列，不计为最终修改。

独立检查从本轮快照提取主 `specification`，逐字比较陈述及证明：449 条保持；没有快照的未改 reader 使用整文件初始哈希一致性。选定 kernel 与 Contract 的 366 项对照均保持。主 reader 的仓库依赖闭包不含 `Misc/`，必需 `Proofs/` 不反向依赖 reader。

```sh
python bench/check_main_entry_roles.py
python bench/check_io_organization.py
python bench/task_progress.py --check
```

第一项检查当前显式 reader 角色与导入闭包；第二项检查逐题覆盖、选定版本、IO/source/evidence 哈希并拦截 reader 中完整 IO record 的回归；第三项校验唯一清单及其渲染。完整 record 的形状检查是保守的，语义是否重复仍以逐题阅读为依据。保护脚本还有未改源码的负例探针：过期 reader／IO／证据 pin、遗漏任务、错误完整 namespace 元数据、完整 reader record 和合法薄实例。结果索引见 [validation.json](validation.json)。

## 明确的验证边界

下列前两项记录 IO 整理时发现的冲突；它们随后已由[独立命名空间修复](../misc-import-fix/README.md)关闭，两题的主入口与全部题级 `Misc/` 入口均通过新的联合导入。原失败证据与本次 IO 审查记录保留。

- `chunk_bwd_dqkg` 的旧 `Misc/Reader.lean` 和 `Misc/HostReader.lean` 各自声明同名 `Reader.realApplication/nativeApplication`，指向不同结果。全可选入口联合导入失败；分别保留每个 reader 的两组 fresh joint 与旧记录 `rfl` 均通过。未更改这些旧名称。
- `matmul_kernel` 的旧 `Misc/Legacy/HostCorrect.lean` 与当前 `Correct.lean` 都声明 `MatmulKernel.correct`。全历史入口联合导入失败；本轮主入口加所有实际受影响的现代辅助入口检查通过。
- `rms_matmul_rbe` 的未改 `Misc/MathCaller.lean` 原无编译对象，且不依赖本轮候选 IO 提取。尝试补编后按限定范围中止；fresh joint 覆盖主入口与其余所选辅助入口，包括实际受影响的 `CandidateEquiv`。不宣称该旧可选入口已重编通过。
- 额外重编未改 safety／model helper 暴露过旧对象代际差异或资源限制。最终局部成功、可选调整撤回、未纳入的旧源重放与失败尝试分别保存在原队列证据中；没有修改共享数值模型或用新数值假设绕过它们。

这次没有进行 GPU 实测、独立全源码回放或 183 题统一导入。已有全库共享代兼容问题与上述可选入口冲突见 [bench/TODO.md](../../../TODO.md)。不能从逐题 scoped joint 通过推断全库统一构建通过。

唯一完成清单仍为 182 题完成、1 题带数值延期完成（`matmul_leakyrelu`）；当前 FP 依赖分类仍为 178 题严格／结构、4 题使用已有实验准入规则、1 题延期。本轮未增加实验义务，也未新增整题完成数。
