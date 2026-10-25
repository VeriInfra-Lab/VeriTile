# Dropout 原始函数体与完整证明重放

两道题的既有 Python 均不变。dropout 保留八条语句，seeded dropout
针对 i32/i64/u64 分别核对完整九条语句；dtype、调用实参和 constexpr
全部在体外。去掉历史 Lean 的额外寄存器写入与 output 自 cast，
保留原始 mask、RNG、阈值、eager division、运算顺序和写回。

实际源码均重新证明执行、安全、独立 Real 公式、FP 完整程序对、
全部输出和 frame。历史骨架明确标注，以真实的完整执行/内存连接
复用；不把任意 backend 的 self cast 当恒等。旧字/字节、物理、
partial-domain、host 和 Tensor 范围保留。共享前端补齐外部种子 ABI
与 FP32 比较参数精度，并保留越界失败；恒等元数据不污染最终值。
add_value 仅显式展开 blockSize=1024，原有限域与内存等价命题不变。

1677 根、本地和独立无项目缓存副本完整构建与联合导入通过；精确
jobs、Git 源码闭包和各项日志见 review.json。仅复用固定外部依赖。
48 项针对性 Python 通过，全量 293 项仅旧 Structural.lean 数值凭据
哈希错误。实际 comparator 因主机隔离不可用，CPU 缺 torch。
无 GPU 实验、数值新规则或新 axiom。48/183 tasks 与 19/19 examples；
173 对中 168 对为函数体、5 对旧 AST。剩余 132+1+2 题继续处理，
草稿仍不合入。
