# RWKV6 原始函数体与完整证明

10 个 original/candidate 对应项完整保留原 Python 的 16 条语句、cast、mask、反向指针、可选状态与全部 scale ABI（包括整数 1 特化）。候选只提前三个 u 相关赋值；完整 Python AST 已核对。原始 Python、输入 Contract 和原 RWKV IO 定义保留，新增调用绑定与完整 Runtime IO。

共享前端修复了有符号指针、嵌套 FP load 和运行整数转换。由此暴露的旧地址和 RMS 转换证明已修复；完整内存等价取代不成立的原始 FlattenOk 声明。没有增加输入限制、任意 cast 恒等式或整 kernel 假设。各题差异详见 review.json。

4846 个累计模块及联合导入通过，包括独立空项目缓存重建；6297 个本地 Lean 源文件与 Git 快照逐字节一致。1010 个定义模块的 18817 个定理（含 private helper）通过标准公理依赖审计。公开 RWKV FP 比较不使用数值假设。

271 项针对性 Python 检查通过；全量 516 项保留一条旧 Structural 数值凭据错误。164 次官方 comparator 检查因主机隔离能力不可用，Torch 检查因缺依赖不可用；均未计通过。没有运行 GPU 或新数值实验。中间失败和停止记录保留。

累计 tasks 110/183、examples 19/19、375 个对应项，73 个任务仍待迁移。保持草稿，等待人工审查后再决定合入。
