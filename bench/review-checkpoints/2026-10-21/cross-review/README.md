# TritonBench-G 第二轮独立复查（2026-10-21）

16 个审查组交换上一轮负责的题目，重新检查全部 **183 个当前选定版本**。
逐题读取实际 Python、输入契约、共享 typed kernel、独立 Real 公式、源码执行与
IO 连接、实际 FP 程序对，以及所用规则的精度、适用域和证据。
上一轮结论只作背景，没有代替本轮源码检查。

**本轮未发现新增的实质源码、数学结论或数值证据问题。完成数保持 76，待完成 107。**
76 题中，74 题依靠严格证明或程序结构，2 题使用已经准入的 FP32 DIV-MUL-RCP
规则（`apply_penalty`、`softmax_triton1`）。当前主证明数值实验延期仍为 0。
107 题的具体程序连接、比较目标等缺口继续列在
[唯一逐题清单](../../../tritonbench_g/task_progress.md)，没有以数值假设延期。
这里的计数针对当前选定版本；原始 benchmark 的缺陷与历史验收另行保留。

本轮修正了三处主入口中的兼容辅助定理归属，完整名称、类型和证明均保留：

| 题目 | 移入的辅助结果 | 当前文件 |
| --- | --- | --- |
| chunk_cumsum_vector | `FPEquiv.memory_io_equiv` | `Misc/KernelFPInterface.lean` |
| l2_norm_triton1 | `FPEquiv.source_io_equivalence` | `Misc/SourceEquivalence.lean` |
| ksoftmax_triton | `FPEquiv.reversed_schedule_io_equivalent` | 选定 grid_y_checked 版本的 `Misc/Native.lean` |

主 `specification` 直接应用原有 `Proofs/` 定理；主等价结论、比较程序、输入域及
数值规则均未改变。同步修正这三题以及 block_sparse_attn、rowwise_quantization_triton
的 README 角色说明，并在清单中补正 triton_matmul 的实际 README 位置。
格式整理没有增加整题完成数。

本轮特别检查了输入前提是否包含待证输出或运行成功、公式是否独立于程序执行、
证明是否连接了候选而非选定源码、分支与尾部掩码是否覆盖，以及量化整数、转换、
累加与 FMA 顺序是否保留。此前记录的 `matmul_leakyrelu` 原源码 Real IO 连接缺口、
`int_scaled_matmul` 独立 Real 缩放连接缺口，以及 int8_matmul_kernel 原程序与
修复程序的 satfinite 反例均继续如实保留；不是新的数值延期。

验证范围如下：

- 183 题分组报告覆盖无遗漏、无重复；当前选定 Python 均可解析。
- 当前全部主入口递归源码依赖不含 `Misc/`，没有未完成证明占位。
  三处迁移后，选定 Correct/FPEquiv 中没有遗留的普通兼容 `theorem`。
- 未改动的 180 题复用上一轮入口检查；重新核对源码、入口对象及检查证据哈希。
  本轮编辑造成的差异单独记录，没有覆盖历史报告或冒称重新构建。
- 六个修改过的 Lean 模块定向构建成功；ksoftmax 的直接 TensorBridge 消费者
  也已重编。三题新的主入口与跟踪辅助入口联合检查全部通过，核对了实际主声明、
  FP 等价结论类型与公理依赖。新增跟踪两份既有 Misc 文件，总数为 360。
- chunk_cumsum_vector、l2_norm_triton1 的新对象位于
  `.veritile-cache/cross-review-20261021/group01-build/lib/lean`，本轮检查明确优先
  使用该路径。旧对象保留；两题的旧对象不被当作修改后源码的构建产物。

本轮复用了适用的内部编译证明，没有进行独立全源码重放、GPU 实测或数值实验。
既有 baseline receipt 缺口仍按原证据范围保留。历史可选子图中的
`max_reduction/Proofs/LegacyFPEquiv → Misc/ASTHost` 归属整理另记为非阻塞事项，
它不在当前主入口闭包内；此前延期的运行时 grid 问题仍在原 TODO 中，没有逐题扩张。

[逐题审查索引与验证证据](audit.json) 引用本轮 16 份独立报告、最终哈希核对及
三题新入口检查。详细公式、连接、条件与阻碍保留在对应报告中，没有复制历史回执树。
