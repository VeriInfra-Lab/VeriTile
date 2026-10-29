# Persistent matmul 原始函数体

四个 FP16/E4M3 原始/候选函数体保留全部 cast、元组赋值、Boolean ceil、persistent tile/K 循环、clamped 指针、live FP32 dot 和输出 dtype 分支。候选只交换 row-index 乘法操作数，完整 Python 模块 AST 已检查；原 Python、公开 Real/FP 规格和契约不变。

48 个任务/回归入口及联合导入、716 个定理的开发检查通过。实际执行、安全、全部 grid 输出和其他内存保持不变重新连接到独立矩阵乘法规格及完整 FP 源程序对。累计 5572 个模块与联合导入通过热构建和空项目缓存重建；7026 个本地 Lean 源文件与 Git 一致；1855 个模块中的 37290 个定理依赖通过标准公理审计。

347 项针对性 Python 测试通过。全量 592 项运行有一条旧 Structural 数值凭据导入错误，未虚假更新 hash。50 次官方 comparator 因主机隔离不可用；Torch/Triton 缺失；未运行 GPU 或新数值实验。失败草稿及日志保留。

累计 tasks 117/183、examples 19/19、413 个原始函数体对应项，66 个任务待迁移。保持草稿，等待人工审查。
