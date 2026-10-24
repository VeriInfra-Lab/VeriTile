# 原样函数体迁移：字节拷贝和三组 worked examples

本批修复保留在修复分支，**用户查看并批准之前不得合入**。全量迁移尚未完成。

| 范围 | Python / Lean 对照 | 验证 |
| --- | --- | --- |
| `cache_transform_triton` 原程序 | 选定文件 `cache_transform_triton_repaired.py` 的 `gather_cache_bytes` / `Kernels.body` | Python 原文件不变；函数体 AST 相同；与旧完整程序相等；公开 Real/FP 重放通过 |
| 同题已有地址优化候选 | 新增 `cache_transform_triton_candidate.py` / `Kernels.quotientRemainderBody` | 明确是已有 Lean 候选的 native 配套；完整程序相等；完整 grid、输出和 frame 证明保留 |
| `VectorAdd` | `native_kernels.py` / `addKernel`、`optimizedKernel` | 新编写的 native 配套；两个完整程序相等；公开 Real/FP 通过 |
| `FlatVectorAdd` | `native_kernels.py` / `addKernelMasked`、`optimizedKernel` | 新编写的 native 配套；两个完整程序相等；公开 Real/FP 通过 |
| `TritonBenchVectorAddition` | `native_kernels.py` / `originalKernel`、`optimizedKernel` | 新编写的 native 配套；两个完整程序相等；公开 Real/FP 通过 |

三个 examples 原来是 Lean 编写的示例。新增 Python 配套保留它们原有的
constexpr 尺寸特化；**不能据此认定 benchmark 原版 Python 的 runtime ABI
已经匹配**。尤其选定的 `vector_addition` task 仍然待迁移。来源区别保留在
`source-references.json`、全量清单和 Python 文件说明中。

函数体没有 `tl.region`、`tl.param`、`$(...)` 或 `tl.load(dtype=...)`。
调用签名在外层给出类型和 constexpr 名称。原本的 cast、mask、指令顺序均保留。
八个旧/新完整 `ComputeKernel` 相等结论都在各自 `Proofs/SourcePreservation.lean`
中由 `rfl` 证明，包含端口及全部语句。旧快照仅用于证明保持性；当前
`Kernels.lean` 不反向导入它们，公开执行证明仍执行当前源码。

共享前端修复了三处问题：

- 整数张量分支中的 Python 整数字面量可直接定型，例如
  `tl.where(active, idx, 0)`；不把浮点字面量或运行表达式改成整数。
- 跟踪 load、寄存器复制和 FP32 算术的源类型。同类型写回保留 identity；
  跨类型或来源未知时保留目的类型转换，避免给 uint8 原样拷贝增加取模操作。
  重新赋值、算术、分支会相应更新或清除已知类型。
- 输入端口扫描能看见组合 load/cast 语法中的内存读取；端口按外部签名排列。
  用于扫描的语法视图不作为执行程序；完整程序相等证明核对了最终结果。

验证记录：

- `integration.log.gz`：8647 jobs 成功，包括本批八个公开入口、四个保持性模块、
  两个可选 cache bridge，dequantize、argmax、KV cache 公开回归及三个前端测试模块。
- `clean-build.log.gz`：3520 jobs 成功。新检出开始时没有项目构建产物或
  `.veritile-cache`；只复用固定版本的外部 Lean/Mathlib packages。
- `source-closure.json`：230 个仓库内 Lean 模块全部由 Git 跟踪，且与新检出逐字一致。
- `python-tests.log`：25 项针对性回归通过；`pairs.json`：本批八个对应检查通过。
- `joint.log`：公开入口和保持性证明联合导入通过；cache 的 FP assumptions 为 none。
- `guards.json`：主公式、reader、IO、任务记录和源码清单漂移检查通过。
- 官方 comparator 未配置，`comparator.log` 保留原始失败信息；不记作通过。
  本批未运行 GPU、native Python 或新的数值实验。

当前全量状态仍为 `complete=false`：183 个 task 中两个使用新函数体入口，
argmax 是旧 AST 模式，178 个尚无完整对应清单，两个间接源文件尚未解析；
19 组 examples 中三个有明确标注来源的 native 配套，另 16 组待迁移。
`--require-complete` 仍按预期失败；不能把本批通过当作整个仓库已完成。

四个受影响 task 的当前审查记录已更新。先前记录保存在
`previous-selected-reviews.json.gz`，没有改写历史验收证据。
