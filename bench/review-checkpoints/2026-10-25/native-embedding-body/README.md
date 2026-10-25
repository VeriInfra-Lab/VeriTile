# embedding 原始 int64 函数体

两份既有 Python 源程序不变，逐句粘贴并保留四处 int64 转换、
mask 和选择/减法顺序。类型、五个已注解 signed64 参数和四个
constexpr 在体外。直接重证实际 widened 行寄存器、完整 64 次
循环成功/全部输出/外部 frame；Mathlib 整数恒等式连接到同一
数学坐标，不假定 FP cast 恒等。完整状态/失败等价、实际访问
轨迹、全部 payload word、native bounds、grid、物理/Tensor/host
范围均保留，26 个题级模块通过。

共享前端没有改变。2036 根全量本地/联合导入检查集成；独立副本
从零项目对象重建全部 26 个受影响根及其完整依赖。未变前题保留
上一轮完整独立凭据，不声称本轮重新冷构建全部前题。精确 jobs
和 Git 闭包见 review.json。68 项针对性 Python 通过；全量 313
项仅既有 Structural 哈希凭据错误。实际 comparator 因主机隔离
不可用，CPU 缺 torch；均未报通过。无 GPU 或新数值实验。

累计 58/183 tasks、19/19 examples；198 对中 193 raw、5 旧 AST。
剩余 122+1+2 题继续，不合入，等待用户最终审阅。
