# 两道整题完成：l2_norm_bwd、layer_norm_fwd

本批仅补齐这两题的完整双程序证明，当前验收从 **76 完成 / 107 待完成** 更新为
**78 完成 / 105 待完成**。其余 181 题的逐题记录不变。此前全库审查是历史基线，
本批没有重新审计 183 题。

| 题目 | Real 正确性 | FP 双程序结论 | 本批完成的连接 |
| --- | --- | --- | --- |
| `l2_norm_bwd` | `Correct.grid_correctness` | `FPEquiv.grid_equiv`：`original ≡[[]] candidate` | repaired signed64、独立步长源码的三处行地址乘法，连接到完整行网格 |
| `layer_norm_fwd` | `correct` | `index_commuted_equiv`：`original ≡[[]] indexCommuted` | 已有局部源码候选连接到完整 ordinary grid，含所有分支和输出 |

两题的生产 Python、`Contract.lean` 和 `Correct.lean` 本批均未修改。
两题均没有新增数值规则、数值实验或数值延期。索引改写使用整数乘法交换律；
FP32 算子、类型转换、运行时参数、计算顺序和归约顺序均保留。

## l2_norm_bwd

- [生产 Python](../../../repairs/tritonbench_g/l2_norm_bwd/l2_norm_bwd_repaired.py)
- [独立候选 Python](../../../repairs/tritonbench_g/l2_norm_bwd/l2_norm_bwd_index_commuted.py)
- [共享 Kernels](../../../repairs/tritonbench_g/l2_norm_bwd/Kernels.lean)
- [Correct](../../../repairs/tritonbench_g/l2_norm_bwd/Correct.lean)
- [FPEquiv](../../../repairs/tritonbench_g/l2_norm_bwd/FPEquiv.lean)

令 `q = ∑ x² + eps`、`dot = ∑ dy*x`，独立 Real 公式为
`dy/sqrt(q) - dot*x/(q*sqrt(q))`。保留原有 total Real sqrt/division 约定；
把该式解释为通常意义的导数需要 `q > 0`。源码到独立公式的 Mathlib 证明直接复用。

候选将 `X`、`DX`、`DY` 三处 `row * stride_*_row` 改为
`stride_*_row * row`。使用各自独立的行列步长，保留 signed64 行列转换和运行时 FP32 epsilon。
原、候选从同一输入内存执行完整行网格；成功、安全、所有有效 `DX` 输出及两侧完整外部内存不变
均为结论。保留 `N ≤ block`、输出地址唯一性及原有内存窗口条件；没有成功或输出值假设。

主 agent 独立比较完整 Python kernel AST，确认仅三处整数操作数交换；另用 Lean `rfl`
验证共享 quotation 整理前后的原 `template` 和 `source` 定义相同。
运行时 grid 乘积上限的共享 TODO 仍按既定范围延期，不在本批展开。

## layer_norm_fwd

- [选定修复版 Python](../../../repairs/tritonbench_g/layer_norm_fwd_nonzero_tile/layer_norm_fwd_nonzero_tile.py)
- [共享 Kernels 与既有候选](../../../repairs/tritonbench_g/layer_norm_fwd_nonzero_tile/Kernels.lean)
- [Correct](../../../repairs/tritonbench_g/layer_norm_fwd_nonzero_tile/Correct.lean)
- [FPEquiv](../../../repairs/tritonbench_g/layer_norm_fwd_nonzero_tile/FPEquiv.lean)

独立 Real 规格覆盖 rowscale、两路 Philox/dropout、残差、均值/方差、归一化及两路 affine 公式。
本批继续使用已经选定的非零物理 tile 修复；没有增加行数或列数为正的前提。

候选复用已有源码变换，仅将 `X` 初始指针中的 `row * stride.x` 改为 `stride.x * row`。
两侧沿用同一选定 tile、独立行执行和不相交写入合并；成功、实际访问安全、七路有效类型化输出
以及两侧完整外部内存不变都已证明。零行、零列和全部原有 flags 保留。
比较接口只供应源程序的比较原语，不假设目标输出或浮点改写规律。

## 针对性验收

新增助手留在 `Proofs/`。主入口源码依赖闭包均无 `Misc/`；已有 optional native、
Tensor 和历史结果保留，其中 layer norm 的四个 `Misc/` 入口联合导入通过。

修改模块经针对性 Lean 构建；主 agent 对两题重新运行主入口和已登记辅助入口导入检查，
检查真实 FP headline 的类型、Real 标记、主定理公理依赖以及源码/对象稳定性。
内部既有证明对象复用。未进行独立全源码冷回放或 GPU 实测，不据此声称硬件一致性或性能提升。

`l2_norm_bwd` 的可选 `#print_fp_assumptions` 传递诊断达到 4,000,000 heartbeat 上限，
已独立记录并从主构建中分离。真实主定理及其标准公理检查通过；空规则集和三处整数改写
可直接由定理类型及局部证明核查。本批不扩展为诊断器性能修复。

验收记录和冻结交付的路径、哈希见 [audit.json](audit.json)；原始构建和检查证据位于
`.veritile-cache/two-task-completion-20261021/`。历史数值规则与实验记录保持原有证据范围。
