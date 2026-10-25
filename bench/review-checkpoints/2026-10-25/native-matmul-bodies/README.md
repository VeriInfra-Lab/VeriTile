# matmul 原始函数体

两份已有 Python 源程序不变，完整函数体直接粘贴，dtype、九个整数
调用值、三个 constexpr 及端口在体外。保留 live dot、ceil K 循环、
所有 mask 和指针推进、FP16 narrowing；隐式 FP32 存储转换由前端补齐。
完整 ComputeKernel 与旧版在全部 Config/区域下相等，包含所有状态、
失败和接口。纯 host 模型原样移至根目录，源码不反向导入证明。

47 题级模块和 95 前端回归重放；1981 根本地、联合导入和零项目
构建缓存的独立副本全量通过，精确 jobs/源闭包见 review.json。
64 项针对性 Python 通过；全量 309 项仅既有 Structural 哈希凭据
错误。实际 comparator 因主机隔离不可用，CPU 缺 torch；均未报通过。
无 GPU 或新数值实验。

累计 56/183 tasks、19/19 examples；194 对中 189 raw、5 旧 AST。
剩余 124+1+2 题继续，不合入，等待用户最终审阅。
