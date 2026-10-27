# Attention 原始函数体与证明

Kernels.lean 内保留两段完整原 Python 函数体；dtype、运行参数和 constexpr
在函数体外绑定。位置参数块指针、形状乘法、dtype 属性、全部 cast、
单元素边界元组、分支、dot/add 和候选 static_assert 均保留。

完整源码 AST、signed PID/地址、循环执行、访存安全、全 grid 输出/history/frame、
独立 Real 公式、实际 FP 候选等价和既有 native/host/Tensor 证明重新连接并验证。
原 Python/两个 wrapper、Correct/FPEquiv/IO/Contract 与旧 Whole001 文件未改。
没有新增 axiom、整体 kernel/model 前提、浮点恒等假设或输入限制。

46 个定点/前端入口和 3050 个完整入口、联合导入均通过。同一 Git 源码独立从
空项目产物重建，只复用固定外部依赖；未用 .veritile-cache 或其他目录项目产物。
20 项假设审计通过，FP assumptions 为 none。109 项针对性 Python 通过；
354 项全量检查只有已知旧 Structural 数值凭据错误。15 次官方 comparator
实际尝试因宿主隔离不可用；CPU checker 缺 Torch。未计为通过，未跑 GPU。

累计 tasks 85/183、examples 19/19，仍有 98 个任务待完成。
kv_cache_filling/triton_argmax 仅回归重放，未增加 raw body 计数。
保持草稿，等待人工审阅，不合入。
