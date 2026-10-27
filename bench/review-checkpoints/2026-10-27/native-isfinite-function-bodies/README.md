# isfinite 原始函数体与证明

Kernels.lean 保留原始 helper 和四段完整 Python writer 函数体，分别覆盖
原程序/已有整数交换候选及 FP32/FP64 输入。类型、原始 CUDA 导入别名、
调用参数、signed byte-view 坐标和 constexpr 位于函数体外，原 cast 均保留。

全部可执行语句与旧证明视图定义相等；修正了旧 collector 漏记输入 x 的
端口 metadata。任意 algebra 下的完整状态执行、失败和访存安全均定义保持。
Python/wrapper、Correct/FPEquiv/IO/Contract、直接执行、安全、Real、all-word、
native、host 和 Tensor 原证明未改，只重新验证。没有新增数值假设或输入限制。

30 个定点入口、7 个额外独立回归文件、16 项假设审计通过，FP assumptions
为 none。3077 个累计入口、联合导入在同一 Git 源码上 warm 与独立空项目
重建均通过，未借用其他目录项目产物或 .veritile-cache。112 项针对性 Python
通过；357 项全量检查只有已知旧 Structural 数值凭据错误。7 次官方 comparator
实际尝试因宿主隔离不可用；CPU checker 缺 Torch。未计为通过，未跑 GPU。

累计 tasks 86/183、examples 19/19，仍有 97 个任务待完成。
kv_cache_filling/triton_argmax 仅回归重放，未增加 raw body 计数。
保持草稿，等待人工审阅，不合入。
