# Ksoftmax 原始函数体与证明

选定版本的 forward/backward 完整保留原 Python 函数体，包括 docstring、6处 int64 转换、None/字符串 mask 选择、FP16 分支、mask 和隐式 store。dtype、标量和 constexpr 由外部调用提供。公开 Python、Correct/FPEquiv/Contract 和输入域不变；既有 FP 比较仍是行调度反序。

实际执行、安全、输出、frame 和独立实数公式重新连接；九个证明确认修正多余 store cast 后，既有 NativeGraph/Real 目标对任意输入和归约计划不变。原版 backward 索引候选的旧证明缺口改由实际完整解释器等式证明。

116个相关模块、119个前端入口、累计4585个入口和联合导入通过，包括独立空项目重建。3333个 elaborated 任务定理（包括 private helper）与1个前端证明公理审计通过；公开FP比较无数值假设。

245项针对性Python检查通过；全量490项保留旧Structural数值凭据错误。选定wrapper的metadata/launch检查通过；共享Torch检查和119次官方comparator检查因环境不可用而未计通过。未运行GPU或新数值实验。

累计tasks108/183、examples19/19、361个对应项，75个任务待修。保持草稿，不合入。
