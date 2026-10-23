# Selected main-entry role review — 2026-10-22

本轮检查全部 **183 题**当前选定的 `Correct.lean` / `FPEquiv.lean`，解决
`index_select_bwd` 暴露的问题：辅助结论与主规格混放。选定路径来自唯一逐题清单
`bench/tritonbench_g/task_progress.json`；没有把同名旧版本当成当前入口。

共检查 366 个选定入口，解析一处 `FPEquivBound → FPEquiv` 转导入后为 367 个文件。
逐声明判读后保留 **449 个真实主规格**；**11 题整理，172 题无需修改**。
本轮范围内未发现剩余辅助证明结论混入主入口，也未发现主入口传递依赖 `Misc/`。
每题的声明、职责、源码哈希和详细审阅证据见 [audit.json](audit.json)。

## 判定原则

- `Correct.lean` 保留独立 Real 数学规格；`FPEquiv.lean` 保留实际两程序的等价主结论。
- 额外 host、Tensor、native、候选应用、数学推论、源码同一性兼容别名放在 `Misc/`。
  换成 `theorem`、`alias` 或 `abbrev` 不能代替职责分离。
- 主证明必需助手留在 `Proofs/`。必要的公式、IO 和程序视图定义可以留在主入口。
- 不按 `specification` 数量一刀切。独立 forward/backward、producer/consumer、实际
  source branch，以及唯一完整的多阶段主功能均保留。直接覆盖统一 source family 的
  Real 主规格也保留，不为整理目录新增逐版本证明任务。
- 已经隔离在 `Misc/` 的历史规格不属于当前主入口；本轮未批量改写这些历史证据。

## 整理的题目

| 题目 | 整理内容 |
| --- | --- |
| `index_select_bwd` | 额外 Python 多次调用组合应用移至 `Misc/CompositionCorrect.lean`；保留单 kernel Real 主规格。 |
| `matrix_vector_multip` | 单 CTA 等价辅助及程序视图移至 `Misc/TileEquiv.lean`；保留完整 grid 主等价。 |
| `fused_recurrent_delta` | 可选兼容证明模块移至 `Misc/KernelCompatibility.lean`，保留旧路径 shim；主入口去除其导入；两项 host 辅助改普通 theorem。 |
| `llama_ff_triton` | 可选 raw 应用移至 `Misc/RawCorrect.lean`，保留旧路径 shim；主入口去除其导入。 |
| `fused_recurrent_hgrn` | 去除只为 native 应用服务的间接导入，直接导入主证明必需模块。 |
| `uniform_sampling` | 覆盖辅助定理移至 `Misc/Coverage.lean`。 |
| `context_attn_fwd` | 源码同一性兼容别名移至 `Misc/SourceCompatibility.lean`。 |
| `triton_attention` | 梯度数学辅助别名移至 `Misc/RealFacts.lean`。 |
| `quant_transpose_kernel` | 归一化数学辅助别名移至 `Misc/RealCorollaries.lean`。 |
| `geglu_tanh_triton` | 可选 raw native 应用从混装 `Proofs/` 模块移至 `Misc/RawNativeApplication.lean`。 |
| `softmax_triton2` | 可选 Tensor/Object 应用与 accessor 从混装 `Proofs/` 模块移至 `Misc/RealApplication.lean`。 |

原完整声明名与数学陈述保留，已有证明继续复用；覆盖别名改为普通 theorem 的连接也已检查。
移出的入口需要显式导入对应 `Misc/` 文件。
原选定 kernel、候选、输入范围、精度和数值规则没有改变。
本轮只修正组织，不新增或撤回整题证明完成数。

## 验证与复查

11 道修改题的针对性构建和 fresh 主／现有可选入口联合检查均通过。
未修改题不要求重编；部分已启动的针对性检查和 fresh joint 已完成，详细日志明确区分
新检查与既有证据复用。`rms_norm_triton/Correct` 的旧记录存在一个直接依赖对象哈希
差异，本轮如实保留其证据限制，不声称对该题做了 fresh 构建。
`kv_cache_filling` 与 `chunked_cumsum_fwd` 的独立选定 generation 按原范围保留，
未把当前 strict overlay 缺对象误报成定理失败。

入口保护检查：

```sh
python bench/check_main_entry_roles.py
python bench/task_progress.py --check
```

前者核对已审阅的逐声明清单和源码哈希、解析转导入，并检查当前仓库依赖闭包不含
`Misc/`、必需 `Proofs/` 不反向导入 reader。修改入口后必须更新职责审阅，不能仅刷新
哈希自动宣称语义通过。这是组织检查，不是 Lean 独立回放或数值正确性证明。
最终两项检查、修改文件的差异与空白检查均通过；保护脚本的三项拒绝探针也通过。
检查日志、11 题构建证据索引和结果见 [validation.json](validation.json)。

此前记录的跨题统一 generation 联合导入问题仍未解决，见
`bench/tritonbench_g/task_progress.json` 的 `current_cross_task_import_check`
与 `bench/TODO.md`。本轮针对性检查不代表统一全库构建通过。
独立完整源码回放和 GPU 实测本轮均未运行；数值分类与唯一延期保持原状。
