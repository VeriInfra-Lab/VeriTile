# LayerNorm 原始函数体与完整证明

4 个对应项均保留原 Python 的 24 条语句，包括 int64 行号、uint32 seed、显式 rowscale cast、所有分支、双 Philox 流及 7 个输出。候选只交换初始 X 行号与步长的整数乘法。原 Python 与 Contract 未改；调用信息与 N=1 特化在函数体外。

实际有类型程序连接到完整执行、访存安全、grid、内存 frame 和 8 个返回视图；精确 Real 解释经完整逻辑 IO 双射连接独立数学公式。FP 比较不使用数值假设。没有 sorry、新 axiom、任意 cast 恒等式或整 kernel 假设。当前源码证明 profile 为 FP32；历史 compiler/storage artifact 是单独的可选接口。

293 个相关模块及联合导入、5139 个累计模块及联合导入通过，包含独立空项目缓存重建。6577 个本地 Lean 源文件与 Git 逐字节一致；1304 个定义模块的 24193 个定理（含 private）通过标准公理依赖审计。

281 项针对性 Python 测试通过；全量 526 项保留一条旧 Structural 数值凭据错误。294 次官方 comparator 检查因主机隔离能力不可用；Torch 检查缺依赖。均未计通过，没有运行 GPU 或新数值实验。保留中间失败和修复记录。

累计 tasks 111/183、examples 19/19、379 个对应项；72 个任务待迁移。保持草稿，等待人工审查后再决定合入。
