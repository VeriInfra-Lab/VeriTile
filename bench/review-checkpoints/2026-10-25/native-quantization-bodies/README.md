# 四道量化任务原始函数体与证明重验

rowwise_quantization_triton、quantize_copy_kv、quantize_kv_transform、
quantize_kv_copy 的原程序和既有候选共8段函数体逐句核对。存储/调用/
constexpr信息体外；source cast、literal、mask、shape、归约及运算/写回次序保留。
选定原Python和wrapper均未改。后三题Python候选副本明确标为从已有Lean
候选补写；第一题候选Python原先已存在。三题完整程序rfl相等；rowwise按
实际signed地址、隐式存储转换和严格整数编码重证，不新增浮点恒等式。

170题级模块与全部依赖从空项目产物独立重建；2661根warm及联合导入通过。
全部旧Real/FP/grid/safety/Tensor/native/host及输出/帧范围保留，精确计数、
Git源码闭包和实际审计见review.json。87项针对性Python通过，全量332项
仍仅旧Structural数值凭据错误，不虚假重钉。官方比较器所需隔离不可用，
CPU缺Torch；实际失败保留不计通过。无GPU或新数值实验。

累计74/183tasks、19/19examples；231对中226raw、5旧AST，其余109题继续。
草稿不合入，等待用户审阅。
