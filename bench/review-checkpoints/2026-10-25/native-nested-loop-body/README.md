# 三层循环的完整原始函数体与有符号执行证明

原始 Python 不变，逐句保留开头的 int64 cast、三层循环、clamp、
unique/mask、14 次缓存读取、24 次写入与 barrier。新 Python 只对应
原有整数乘法候选。实际有符号寄存器的完整执行、安全及最终地址均
直接证明；旧自然数骨架明确留在 Proofs，并以双方完整执行的成功/
失败和全部内存等式连接。原始/候选完整状态相等，FP 不用实验规则。
全部 59 个题内 Lean 模块、独立 Real 公式、输出/frame、word/byte、
host/Tensor 与历史范围复验。修正旧辅助 Real 定理的一维索引名称，
保留其旧范围，并补当前矩阵坐标公式的 Real 连接。

1625 根、本地及独立无项目缓存副本完整构建和联合导入通过；精确
jobs 与源码闭包见 review.json。仅复用固定外部依赖，无项目 build
或 .veritile-cache。45 项针对性 Python 通过，全量 290 项仅旧
Structural.lean 数值凭据哈希错误。实际 comparator 因主机隔离
不可用；CPU 缺 torch。无 GPU 实验。46/183 tasks、19/19 examples，
165 对中 160 对为函数体、5 对旧 AST。剩余 134+1+2 题待处理，
草稿不合入。
