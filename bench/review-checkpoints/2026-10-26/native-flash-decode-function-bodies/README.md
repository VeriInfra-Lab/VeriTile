# Flash Decode 原始函数体与证明

Llama/Phi 原程序和既有 shared-base CSE 候选共四段完整 13 条顶层语句。
三个 INT64 cast、guarded ceil count、0.0/-inf 初值、signed loop、全部
浮点运算顺序、Phi 两处 mask、最后 store 和 return 都保留。
指针 dtype、真实 H/S/D stride 关系和 constexpr 在函数体外绑定。
原 Python/wrapper、公开 Correct/FPEquiv/IO/Contract 均未改动。

通用前端修正普通浮点常量赋值的强类型追踪，内联常量保持 weak。
保留显式转换和真实输出转换；不支持的 FP64 常量赋值会明确拒绝。
四个 Lean 类型/转换回归和 Python 中的 body/ABI 变异检查均通过。

实际 signed 执行、完整输出和 frame、循环访存安全、保留失败和完整
状态的整数 CSE、两个完整 grid 的 Real 数学规格和 FP 等价、native/
有限索引/host/Tensor 以及原定义域全部重放。没有新增 FP 数值假设。

2948 根 warm/joint/空项目产物独立完整重建通过；未复用其他 checkout
项目产物，也未使用 .veritile-cache。源码 closure 对应同一 Git commit。
105 项针对性 Python 通过；全量 350 项仅旧 Structural 数值凭据错误。
12 次官方 comparator 因宿主隔离不可用，实际 CPU checker 缺 Torch，
保留实际结果，不记通过。选定 Phi Block 没有独立 CPU checker；
历史共享 Phi 检查不冒充选定 wrapper 的检查。无 GPU 或新增数值实验。

当前 examples 19/19、tasks 83/183、raw pairs 250/255（其余 5 为旧 AST）。
剩余 100 个任务继续处理；完整语料迁移未完成。等待人工审阅，禁止合入。
