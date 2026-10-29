# Liger 完整原始与候选函数体

四份函数体直接来自未改动的 layer_norm_liger_repaired.py 和既有 layer_norm_liger_optimized.py。保留每条 cast、mask、指针更新、标量循环步长、归约及输出转换。调用类型和 FP32 dtype constexpr 在 quotation 外，输入域和公开 Real/FP 规格不变。实际执行、独立数学公式、完整原程序/候选 FP pair 与 IO、安全、frame 均重新验证。旧缩略 AST 的诊断等式改为实际执行与完整内存观察关系。

84 个任务/前端模块联合通过，2179 个定理通过依赖审计。累计 6013 个入口通过常规与空项目缓存独立重建；7454 个本地 Lean 文件与 Git 完全一致；2494 个模块的 56005 个定理通过标准公理依赖审计。

419 项针对性 Python 测试通过。全量 664 项保留一条历史 Structural 数值凭据导入错误，旧凭据未重写。89 次实际 comparator 因主机隔离不可用；Torch/Triton 缺失，未进行 GPU 或新数值实验。通用 tl.full 有符号位宽传播仍单独记录。

tasks 122/183、examples 19/19、429 个完整函数体映射，61 个任务待迁移。当前仍是草稿，未合入。
