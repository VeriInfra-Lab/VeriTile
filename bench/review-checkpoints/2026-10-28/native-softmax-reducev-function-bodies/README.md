# SoftmaxReduceV 原始函数体与证明

原程序与既有 early-address 候选都保留完整原始函数体，各18条顶层语句，包括 return。候选仍只提前计算两个输出地址。调用 dtype、运行标量与 constexpr 参数放在函数体之外。

前端跟踪实际 FP32 存储读入和复制，在归约边界投影；普通算术寄存器不增加恒等 cast。初始化和 padding 精度标注修正为原始源码的实际含义，并证明之前 NativeGraph 与 RealModel 的目标结果对任意输入、归约计划保持不变。Python、公开 Correct/FPEquiv/Contract 和输入域不变。

47个本题入口、118个前端入口、累计4468个入口及联合导入通过，含独立空项目重建。165个任务定理和3个前端证明公理审计通过，公开FP比较无数值假设。保留被拒绝的过宽 cast 处理和旧精度标注失败日志。

237项针对性Python检查通过；全量482项保留旧Structural数值凭据错误。51次comparator尝试受隔离限制，CPU checker缺少Torch，均未计通过；未运行GPU或新数值实验。

累计tasks107/183、examples19/19、359个对应项，76个任务待修。保持草稿，不合入。
