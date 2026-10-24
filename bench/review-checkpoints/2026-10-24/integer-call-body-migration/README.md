# 运行参数与原始 Python 函数体迁移

本批新增十三道 task 的原样函数体入口，仍在修复分支和草稿 PR 中。
**用户审阅并明确批准之前不得合入。全量迁移尚未完成。**

| 本批任务 | 核对内容 |
| --- | --- |
| `add_example`、`add_value`、`vector_addition`、`vector_addition_custom` | 原始与已有候选的变量、完整函数体、运行 int32 参数、常量及 mask；完整程序相等 |
| `cosine_compute`、`sin_computation`、`sin_kernel` | 原始 cast、`tl.cos` / `tl.sin` / `tl.math.sin`、运行参数及候选；完整程序相等 |
| `square_matrix`、`matrix_transpose` | 实际运行 stride 参数、原始 FP32 / FP16 存储、转置及候选；完整程序相等 |
| `index_select_bwd` | 完整执行语句与执行语义相等；补正旧扫描器漏掉的两个带 cast 的输入 load 端口 |
| `kldiv_compute` | 原始 `x * tl.log(x / y)` 和候选保持完整程序相等；移除实际源码对 Proofs/HistoricalKernels 的反向依赖 |
| `swiglu_fwd` | 原始 FP32 cast、sigmoid、乘法顺序、指针更新和 `0.` 拼写；通过精确字面量等式连接旧完整程序 |
| `l2_norm_triton1` | 删除旧 Lean 多加的 `eps.to(tl.float32)`，同步执行公式，重新证明完整执行、输出、安全、frame、Real 规格和实际 FP 程序对 |

每题的 `jit-correspondence.json` 明确给出真实 Python、实际 Lean 定义、调用
签名和整数绑定。新写的三个 `candidate_kernels.py` 只是已有 Lean 候选的 native
转录；原版仍对照各自选定的独立 Python 文件。

`TritonBenchVectorAddition` 已改为直接对照 benchmark 原始
`bench/tritonbench_g/vector_addition/vector_addition.py`，候选对照已有 repair。
其尺寸保持原生 runtime int32 ABI，旧的新增镜像文件已删除。
`VectorAdd`、`FlatVectorAdd` 仍明确标为新编写的配套，不冒充历史原版。

共享前端现在保留原始注释、docstring、跨行调用、末尾调用逗号及 Python 缩进。
类型、端口和调用值位于 quotation 外。整数调用绑定支持现有 Nat/Int 索引投影，
不会把 runtime 参数改标为 constexpr；原有 native 范围和 IO 连接义务保留。
FP32/FP64 运行标量仍由实际调用提供有精度的寄存器。不能处理的参数重赋值、
语法或控制流会报错。使用方式见 [函数体入口说明](../../../../docs/triton-body-copy.md)。

验证针对代码提交 `f6bed6cc425e31197f56eafb00203306df3529b3`：

- 公开入口、保持性证明与前端回归的集成构建通过，8,800 jobs。
- 十三道任务和三组 examples 的全部 329 个 Lean 模块通过，8,944 jobs；
  包含 Misc/legacy 辅助模块。四处辅助连接脚本已同步，命题保持不变。
- 独立检出起初没有项目构建产物或 `.veritile-cache`，仅复用固定版本的外部
  packages。先重建公开入口，再检出四处证明脚本修正并重建全部组合目标：
  339 个显式模块、9,029 jobs 通过。没有共享项目 build 目录。
- 782 个仓库内依赖模块全部由 Git 跟踪，并与独立检出逐字相同。
- 公开入口联合导入通过；各模块的实际 FP assumptions 和 axiom audits 保存在日志中。
- 39 个来源对应检查通过，其中 34 个为函数体模式，5 个仍是 argmax 的旧 AST 模式。
- 29 项针对性 Python 回归通过。全量 Python 测试运行 274 项，一项旧数值证据
  哈希检查失败；变更前的干净基线也复现同一失败。不能把全量测试记为通过。
- 官方 comparator 未配置，实际尝试的失败保存在 `comparator.log`，不记为通过。
  没有运行新的 GPU/native 数值实验，也没有新增数值定律或 axioms。

源码清单仍为 `complete=false`：183 个 task 中，15 个完成函数体对应，
1 个为旧 AST 模式，165 个尚待完整对应，2 个间接加载源码尚待入口接线；
19 组 examples 中，1 个对照独立原版，2 个为明确标注的新配套，16 个缺少已登记
的独立 Python 来源。不能把本批证明通过当作整个仓库的迁移完成。

`review.json` 给出逐题语义检查及验证范围；`source-closure.json` 给出完整依赖
源码；当前主公式记录和任务清单已更新，旧记录保存在
`previous-selected-reviews.json.gz`，历史验收证据保持不变。
