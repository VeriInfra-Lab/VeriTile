# 两层完整循环的整数位值证明

`FPCorrect.int8_matmul_word_correctness` 覆盖原始四个 packed field 及任意内层 K 块。结果由初始 A/B 内存的独立递推定义，完整程序成功执行，每个活跃输出都等于该递推，且未写地址不变。没有 no-overflow 假设，没有把最终输出放进前提。

该内核本身不做浮点算术，但旧 `.int`/`.nat` 模型擦除了真实位宽，因此也属于本次数值修复。`WordSource.widths` 补回原源码两次显式 `.to(tl.int8)` 和每次 int32 累加赋值的位宽。int8 转换取低 8 位再按补码解释；点积逐项进行 32 位乘加；每次 accumulator 更新再取低 32 位。`WordBits.dot_inputs_valid` 对所有实际 dot 输入证明 signed-byte 范围。旧无限精度整数结论继续作为历史数学证明，不代替这个结果。

源码保留同一分组 program ID、每次 mask、pointer advance、四个 field 的先后次序和内层循环。A 指针跨两层循环连续前进，B 在每个 field 开头复位。`original_decomposition` 核对完整原 AST 的分段；`annotated_inner_length` 核对全部 9 个内层语句。唯一另一个计算表示调整是将旧 `.alg(if false ...)` 静态断言占位变成同义的 compute `if false`，公开条件仍保留实际 `K = 4*(BK*steps)` 与 `BK > 0`。

公开输入范围包括真实 int32 A、uint8 B 存储以及活跃写地址互不重叠。整数转换和溢出回绕是模型中的定义及证明，不是交给实验验证的规律。`WordWitness.nonempty_word_execution` 对 16×16 输出 tile、BK=32、四个 field 证明完整执行：A 的 255 转成 -1，B 的每个 field 解为 2，输出为 -256；masked row 与 stride gap 保持原值。另有 signed-byte 截断改变乘积、int32 溢出的具体位例。

尚待设备实验绑定的是原编译器的 int8 cast、int8 dot 的 int32 wrap/饱和选择、原始 uint8/int32 load/store、合法索引位宽与启动配置。没有运行或宣称通过 GPU 实验。软件位证明不要求额外的结合律、实数等式或经验舍入规律。
