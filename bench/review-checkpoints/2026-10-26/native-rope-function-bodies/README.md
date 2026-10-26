# RoPE 前/反向原始函数体及直接证明

原 Python JIT、wrapper、公开 Correct/FPEquiv/IO/Contract 不改。
两题各原程序/已有索引 CSE 候选，四段完整函数体，各 23 条顶层语句。
指针类型、运行 stride/sequence、constexpr 均在体外。保留所有 dtype cast、
padding、两分支和 store；反向独立 cos/sin 列 stride 保留，不复用前向假设。
候选 Python 从既有四处坐标/mask CSE 补写并明确标注。

前向 ActualSource 直接重证实际原始函数体的完整执行及独立有序内存公式，
历史 typed/algorithm 源及旧 AST 审计保持逐字不变。旧 erasure helper 明确
标为 historical，不把当前保留 typed cast/constexpr 商的 AST 冒充旧 AST。
当前实际 Real/FP/安全结论通过直接执行、值、mask 与完整 grid/frame 连接。
反向证明直接引用 canonical raw source 并重证，不维护第二份语句列表。

83 个题级根加 9 个历史辅助根共 92 根，从空项目产物及完整依赖独立重建；
2827 根 warm、联合导入；所有本地 Lean 源码与 Git 匹配。
100 项针对性 Python 通过，全量 345 项仅旧 Structural 数值凭据错误。
官方比较器所需隔离不可用，两个实际 CPU probe 缺 Torch；失败保留不计通过。
无新 FP 假设、输入限制、GPU 或新数值实验。

累计 80/183 tasks、19/19 examples；249 对中 244 raw、5 旧 AST。
仍有 103 题待迁移。保留草稿，不合入，等待用户审阅。
