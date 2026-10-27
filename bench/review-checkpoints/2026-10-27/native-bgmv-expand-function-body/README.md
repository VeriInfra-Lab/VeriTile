# BGMV Expand 原始函数体与证明

四段完整函数体覆盖原程序、既有 output-first 候选及两种 metadata ABI。
保留 docstring、所有 cast、提前 return、分支、signed cdiv/range、mask、
half 乘法/归约及 store。dtype、连续 stride、constexpr 均在函数体外绑定。
原 Python/wrapper 与公开 Correct/FPEquiv/Contract 未改；IO 有序公式去掉
原先虚构的 FP32 同格式转换，保留真实的 half→FP32 转换与全部布局/定义域。

完整源码与证明视图的状态/失败/安全连接、实际执行、全 grid 输出/frame、
访存安全、独立 Real 公式、FP 候选等价及 native/host/Tensor 证明均重放。
共享前端修复后，其他受影响的索引状态、地址与执行连接证明也已更新。
没有新增 axiom、整体 kernel/model 前提、浮点 cast 恒等假设或输入限制。

112 个定点入口、3008 个完整入口及联合导入通过；同一 Git 源码另在空项目
编译产物目录完整重建。仅复用固定外部依赖，无 .veritile-cache、无跨目录
项目产物复用。18 项独立假设审计通过，FP assumptions 为 none。
107 项针对性 Python 通过；352 项全量检查只有已知旧 Structural 数值凭据错误。
65 次官方 comparator 实际尝试因宿主隔离不可用；CPU checker 缺 Torch。
这两项不计通过，没有运行 GPU 或新增数值实验。

当前 tasks 84/183、examples 19/19；仍有 99 个任务待完成。
kv_cache_filling/triton_argmax 本批仅重放回归，未增加函数体迁移计数。
完整语料尚未完成，保持草稿，等待人工审阅，不合入。
