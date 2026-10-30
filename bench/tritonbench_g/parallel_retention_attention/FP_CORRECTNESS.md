# Retention 有限精度契约

`FPCorrect.lean` 的两个公开声明完整覆盖此前接受的逐 program forward O 和 backward DK/DV 阶段，均为 `backend_value_contract`。两段循环、实际迭代顺序、中间衰减、输出 cast/store 和全部内存 frame 都已证明；没有缩为首块。

`forward_fp32` / `backward_fp32` 从完整 fp32 storage 源程序证明执行成功，且全部最终内存等于独立逐原语值递推的写回结果。独立规格分别位于 `FPForwardGraph.lean` 和 `FPBackwardGraph.lean`，不调用源执行函数或假设最终结果。它们保留 fp32 sub/mul/add、literal、integer conversion、exp2、log2、每个 dot 的 `allow_tf32=false`、cast、mask 和未指定 padding 的 undef 输入。

修复了旧数学转写的两个差异：实际常量 `-5` 以 fp32 标量 literal 表示；diagonal 的负索引差采用有符号整数差。`tl.full([], …, dtype=tl.float32)` 只显式标注 Python/Triton 浮点常量的默认精度。原生正常 block 范围内的索引需满足无溢出条件，才能把此处加宽有符号整数表示绑定到原生 int32 运算。旧数学端的 clipped Nat 差和 `0-5` 没有冒充新浮点证明。`FPShape.lean` 证明修正后的完整源码可降低到算法层，未声称它与旧错误树字面相等。

精确原生绑定仍需完成：

- `M.unary (some fp32) exp2` 和 `log2` 的实际实现、特殊值和 subnormal 行为。这里没有使用 Real.exp/log，也没有闭合的 exp2/log2 bit interpreter。
- `M.fromNat` / `M.fromInt` 的 fp32 转换；fp32 常量、load、cast、sub/mul/add 的 rounding/FTZ、NaN/Inf 和 signed-zero 策略。
- 每个 dot 的真实累积树/accumulator/FMA/options。`allow_tf32=false` 被保留，但不能单独证明指定软件 dot 与 GPU 一致。
- 普通 mul/add 是否被 compiler fusion；合法 block/grid/stride、地址和整数范围。正常 host forward BTL=128/BTS=32，backward BTL=64/BTS=32，num_warps=4。

上述参数是原语语义和实现对应，不是额外实数代数律、round=id 或最终输出假设。所有源码尾部 undef 保留；因此证明不把含未定义读的输出变成确定数值。输出地址别名时也保留实际 store 顺序。

DQ、host NK/NV 合并、autograd wrapper、跨 program 并发和 GPU 原生对应仍不在此前接受阶段范围。此次没有宣称闭合 bit interpreter、真实 GPU 正确性或这些外层功能已完成。

原生实验 receipt 应包含源码/声明/profile hash、编译器及编译选项/产物、GPU 型号、输入输出位模式、明确采样范围与反例；抽样吻合不能普遍证明硬件原语对应或恒等舍入。
