# 累计求和与 logsumexp 原始函数体

两题既有原始及候选 Python 全部不变，完整函数体直接粘贴。
累计求和保留 signed64 边界、全部 cast、未指定尾部、scan/sum/
carry 和元素类型写回；体外调用保持旧 FP32 存储通道。完整程序
AST 相等覆盖此前所有泛型 dtype 模型，不假定 cast 恒等。

logsumexp 保留 float("inf")、实际 FP32 scale、两个分支与 mask，
前端复用精确无穷字值原语。完整可失败状态执行连接旧模型，
公开 Real、FP 程序对和全部旧 native/host/Tensor 范围均重放。
前端补齐属性 cast、整数 literal/range、并行赋值和 max 类型跟踪；
保持原有循环不变量与 RNG ABI 拒绝检查，无额外源码语句或数值规则。

1779 根本地、联合导入与独立无项目缓存副本完整重建通过，
仅复用固定外部依赖；精确 jobs、Git 闭包与日志见 review.json。
95 前端回归、74 题级模块、55 项针对性 Python 通过；全量 300 项
仅原有 Structural.lean 凭据哈希错误。实际 comparator 因主机隔离
不可用，CPU 缺 torch；未跑 GPU。51/183 tasks、19/19 examples，
183 对中 178 为函数体，5 为旧 AST；剩余 129+1+2 题，继续且不合入。
