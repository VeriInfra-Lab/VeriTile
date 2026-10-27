# argmax 原始函数体与证明

三段原程序和两段已有候选函数体直接保留在 Kernels/CandidateKernels 中；原 Python、
wrapper、公开规格和历史投影源码未改。类型和调用信息在函数体外。保留所有 cast、
constexpr、dtype 查询、NaN/最早索引选择、clamp、归约、重读、循环、mask 与 store。

15 种源 dtype 的完整原/候选结构执行、失败和安全连接通过；native global IO
保持原有 13 种类型范围。旧转写多加的 FP32 同型 cast 已移除，实际浮点执行重新
证明；没有假设任意 cast 恒等。数学、native、完整 grid 和访问安全证明重新验证。

首次累计构建暴露了 constexpr 幂表达式解析回归；原失败日志和修复提交均保留。
最终源码的 3249 个累计入口及联合导入在 warm 和独立空项目构建中通过。25 项
假设审计通过，两个公开 FP assumptions 均为 none。117 项针对性 Python 通过；
362 项全量检查只有已知旧 Structural 数值凭据错误。26 次官方 comparator 因宿主
隔离不可用，CPU checker 缺 Torch，均如实记录。未跑 GPU 或新数值实验。

累计 tasks 88/183、examples 19/19，仍有 95 个任务待完成。保持草稿，等待人工审阅。
