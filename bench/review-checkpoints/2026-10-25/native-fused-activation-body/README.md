# fused_activation 原始字符串分支

两份已有 Python 源程序不变，完整函数体直接粘贴；保留 String 比较、
elif、缺省无 else、切片、访存选项和运算顺序。类型/constexpr 在
体外，已有 Bool 调用 API 提供两个有效 String 值，不改源码条件。
通用前端依据实际调用的有限字符串备选值和之前同名测试化简分支，
再跟踪 dtype；拒绝不支持的 opaque/mutable String。

完整 ComputeKernel 在两个调用选择下与旧版相等，包含所有端口、
状态、失败和轨迹。Real 入口显式应用 Projection 后接回独立数学
证明。29 题级模块、95 前端回归及前题 matmul 47 模块均重放。
2010 根本地、联合导入和零项目构建缓存的独立副本全量通过，精确
jobs 和 Git 源闭包见 review.json。66 项针对性 Python 通过；全量
311 项仅既有 Structural 哈希凭据错误。实际 comparator 因主机隔离
不可用，CPU 缺 torch；均未报通过。无 GPU 或新数值实验。

累计 57/183 tasks、19/19 examples；196 对中 191 raw、5 旧 AST。
首次冻结的字符串类型探测暴露未定数字类型被误解为 String 的回归；
旧构建已中止，记录保留。修正为只读取已确定类型后，完整 kernel 的
数字常量回归和独立复现检查通过，最终全量记录仅接受修正冻结。

剩余 123+1+2 题继续，不合入，等待用户最终审阅。
