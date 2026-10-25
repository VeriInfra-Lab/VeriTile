# chunked_cumsum_fwd 原始函数体

原始 Python 不变，完整函数体直接粘贴，17 个整数调用值、两个 FP32
运行时 clamp、四个 constexpr、输入类型和内存角色均在体外。保留
全部 uint64 cast、bias/softplus、mask、min/max/remask、两次写回和 scan。
既有 scaleFirst 候选的新 Python 对应明确标注来源。完整语句体 AST
及可失败执行状态与旧模型相等；不假定 FP cast 恒等。

两个旧 FP32 typed 输入和三个 profiled 通道分别保留。前端 cumsum
类型跟踪消除误加的 FP32 self-cast，真实 FP16 narrowing 不删。旧的
任意后端比较模型明确归档并保留所有名字、原比较 AST 和前提；当前
实际源码仍用独立 minimum 原语，不假定浮点 min 等于 compare/select。
Mathlib 证明两种实数投影在所有状态（含失败/未定义单元）下求值相同。

61 题级模块、全部 95 前端回归和此前迁移范围均重放。1934 根本地、
联合导入及零项目构建缓存的独立副本全量通过，精确 jobs 与 Git 闭包
见 review.json；只复用固定外部依赖。62 项针对性 Python 通过；全量
307 项仅既有 Structural.lean 凭据哈希错误。实际 comparator 因主机
隔离不可用，CPU 缺 torch；没有 GPU 或新数值实验。

累计 55/183 tasks、19/19 examples；192 对中 187 raw、5 旧 AST。
剩余 125+1+2 题继续，不合入，等待用户最终审阅。
