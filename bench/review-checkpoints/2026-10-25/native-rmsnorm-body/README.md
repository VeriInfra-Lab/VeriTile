# RMSNorm 原始函数体与完整证明重放

原始 Python 不变，四份十一语句源码覆盖实际运行时和特化 epsilon，
保留乘法后、权重前的显式输出类型 cast；调用类型与实参在体外。
ActualRow/Grid 重证完整执行、输出及 frame；实际运行时 epsilon 字值
连接独立 native 公式。唯一 IO 同时绑定两程序，完整状态等价包括失败。
历史解释通过明确的体外 ABI 参数与全部内存观测证明连接；旧 Real、
native、物理、host/Tensor 范围保留，无任意 backend cast 恒等假设。

共享前端识别指针元素类型 cast，避免多余 store 转换。fused 版本
检查全部既有运行时/特化接口并移除反向 Proofs 导入。
1704 根、本地和独立无项目缓存副本完整构建、联合导入通过。
50 项针对性 Python 通过，全量 295 项仅既有 Structural.lean 数值
凭据哈希错误。实际 comparator 的主机隔离不可用，CPU 缺 torch。
未运行 GPU，无新数值规则或 axiom。49/183 tasks、19/19 examples，
179 对中 174 为函数体、5 为旧 AST；剩余 131+1+2 题，仍不合入。
