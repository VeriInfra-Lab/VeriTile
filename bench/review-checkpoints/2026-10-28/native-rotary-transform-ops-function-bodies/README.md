# RotaryTransformOps 原始函数体与证明

两个原始 Python kernel 均保持不变，完整函数体和十二种调用配置逐句对应。既有候选仅把相邻索引计算改成 xor。数学 Model 与迁移前定义逐字节相同，公开输入域、width≤256、完整 grid、FP32 编码、suffix/frame、aliasing 和空输入条件均保留。完整源码执行与安全性接回 Real、FP 及原有 native/Tensor 结论。

64 个本题入口、114 个前端入口、累计 4364 个入口及联合导入通过；301 个定理公理审计通过，公开 FP 比较无数值假设。本次共享编译器未变，独立空项目重建覆盖本题 64 个入口及完整导入闭包，其他历史入口由累计 warm replay 覆盖。

220 项针对性 Python 检查通过；全量 465 项保留既有 Structural 凭据错误。64 次官方 comparator 尝试受隔离限制，CPU checker 缺少 Torch，均未计为通过；未运行 GPU 或新数值实验。

累计 tasks105/183、examples19/19、350 个对应项，78 个任务待修。保持草稿，不合入。
