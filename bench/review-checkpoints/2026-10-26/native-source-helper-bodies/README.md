# 原始 ReLU / SwiGLU 函数体与辅助函数

原始 Python JIT、helper、wrapper 均不改。2 个 helper 与 6 个完整 writer
共 8 对 raw body；类型、运行绑定、constexpr、helper 注册、指针原点均在体外。
逐句保留分支/循环、INT64 cast、block-pointer、边界检查、访存和浮点顺序。
候选 Python 依据既有 Lean 候选补写并明确标注。

ReLU 重证完整分支/循环和有符号视图、实际 FP32 字面量、Real/native、
FP 失败/成功等价、safety/Tensor/host/frame；SwiGLU 以 4 个完整程序恒等
连接原有前/反向证明。共 58 个题级根，公开输入域与规格不缩小，无新 FP 假设。
共享 parser 首次全量重放暴露的 comma token 与旧 DSL 字面量问题已修正；
初始失败及修复日志均保留，最终凭据仅针对修正后的 Git 源码。

403 个针对性根通过；2703 根及全部依赖 warm、联合导入及空项目独立重建。
96 项针对性 Python 通过；全量 341 项仅旧 Structural 数值凭据错误。
官方比较器所需隔离不可用，CPU probe 缺 Torch；实际失败不计通过。
源码闭包、具体 jobs 与逐题核对说明见 review.json。无 GPU/新数值实验。

累计 77/183 tasks、19/19 examples；243 对中 238 raw、5 旧 AST。
仍有 106 题待迁移；本记录不表示全任务完成。草稿不合入，等待用户审阅。
