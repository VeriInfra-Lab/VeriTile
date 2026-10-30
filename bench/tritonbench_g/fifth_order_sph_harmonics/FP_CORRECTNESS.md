# 完整 forward 的浮点值证明

`FPCorrect.lean` 的两个公开规格覆盖原 `fifth_order_fwd` 的全部 11 个输出通道：完整执行成功，从初始坐标内存计算的独立输出规格，以及所有未写地址保持原值。`TypedExecution`、`TypedContract`、`TypedReadback` 分别负责执行、完整写集合、活跃通道读回。

范围是 FP32 坐标和输出、`outputStride > 10`，以及每个活跃输出行的三个输入坐标均有效（`InputCover`）。这些是布局与输入范围条件；不假设最终答案，不要求浮点结合律或实数与浮点相等。先读取所有坐标再写回，允许输入与输出区域重合。反向内核不属于此前接受的 forward 范围。

`TypedReference.channels` 按 Python 中每次乘、加、减、取负的原顺序定义结果。`TypedSoftware.bitChannels` 使用二进制32位运算解释器；25 个不同十进制系数的舍入结果全部由 `coefficient_words` 在 Lean 内核检查。数据通路没有留给外部的浮点 primitive。`rest` 只填充执行接口中未使用的操作；有效输出完全由输入位和 `cfg` 决定。

`Proofs/check_fp_source.py` 将完整 Python forward 的赋值、索引、mask 和 11 次 store 与 typed DSL 逐 AST 对照。只消除显式 FP32 区域声明以及系数的 FP32 类型标注，不做代数变形。特别保留 Y07 的减法和负的 CONST031，以及 Y09 原 CONST020；旧实数证明里的等价改写不能用于浮点证明。

本证明的编译配置采用分别舍入的乘/加。仍需用户 GPU 实验绑定：系数 lowering、未融合的运算顺序、FP32 rounding/FTZ/NaN 行为、加载与存储位保持。启用 FMA 或编译器重结合的配置不能直接沿用这个结果。`Proofs/check_fp_source.py` 和软件模型通过，不代表已经执行 GPU 对照实验。
