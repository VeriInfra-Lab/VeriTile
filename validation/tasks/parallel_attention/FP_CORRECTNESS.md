# PA 有限精度契约

`FPCorrect.lean` 完整覆盖此前接受的逐 program forward O/Z 及 backward DK/DV 阶段。两段循环都保留任意迭代数，没有缩为首块。此次没有改写旧的数学证明。

| 声明 | 已证明的内容 | 模型 |
|---|---|---|
| `forward_fp32` | 原 forward 两循环成功执行；O/Z 独立逐算子递推；精确全部内存写回和 frame | backend_value_contract |
| `backward_fp32` | 原 DK/DV 两循环成功执行；独立逐算子递推；精确全部内存写回和 frame | backend_value_contract |
| `forward_software_fp32` | scale=1 时完整 forward 的独立 FP.Value/ScalarOps 递推 | bit_interpreter_contract |
| `forward_nonempty_rounding` | 两循环均实际执行，输出 `1 + 2^-25` 舍入为 `1`，并保留未写单元 | bit_interpreter_contract |

`Proofs/FPSources.lean` 是 fp32 storage 配置的源程序。每次 load/cast、两次 dot、score 平方、sum、add、masked store 以及 `allow_tf32=false` 均进入 FP.Structural。未指定 padding 的尾部继续读取 `State.undef`，没有补成 0。反向两循环的缩放/相加/掩码顺序分别建模，没有用分配律或结合律化简。`FPShape.lean` 证明新类型标注与原接受源码的 eraseDType 结构一致。

`FPForwardGraph.lean` / `FPBackwardGraph.lean` 的值规格直接定义输入地址、浮点原语调用和顺序递推，不引用源执行函数或最终执行结果。证明没有额外数值代数假设。`DotPrimitive` 是完整 dot 原语的参数化实现（包含矩阵 shape、precision、options），不是已证明的硬件结果。精确全内存等式同时保留输出地址别名时的实际 store 顺序。

软件 profile 使用正零初始化、按 K 升序逐项 `FP.mul` 后 `FP.add` 的 dot，以及按轴升序逐项 `FP.add` 的 sum；每步是真实 binary32 舍入，支持传入的 FP.Config。完整软件契约的 `scale=1` 关闭所有使用中的原语对 `rest` 的依赖。非空见证是 BTL=BTS=BK=BV=1、T=2 的解释器实例，两个时间块均实际执行；unit dot tile 不是合法 CUDA 启动的证明。

尚需绑定到原生执行的内容：

- fp32 load、cast、literal、mul/add，以及数值 domain（NaN/Inf、signed zero、subnormal、payload 策略）；给出目标架构、编译器版本、rounding 和 FTZ 配置。
- 实际 dot 的精确 K 累积树、accumulator 类型、FMA 方式、`allow_tf32=false` 对应。这个选项本身不能证明软件顺序 dot 与硬件一致。
- 实际 `tl.sum` 的归约树，以及普通乘加是否被 compiler fusion。源级分离 mul/add 若与原生融合不同，必须提供正确的绑定或禁用对应 fusion。
- 合法原生 block/grid/stride 与地址/索引范围。host 默认 forward BTL=128、BTS=32，backward BTL=64、BTS=32；BK/BV 经 host 规则选择，至少 16。

源码 proof 对所有保留的 undef 值成立，但不意味着有尾部时输出数值已确定。DQ、host NK/NV 合并、最终归一化、autograd wrapper、跨 program 并发和 GPU 原生对应不在此前接受阶段内；此次也没有声称覆盖它们。

以上原生绑定是实现对应工作，不是 `round=id`、结合律或“最终输出相等”假设。实验应保存精确 source/statement/profile hash、编译选项和产物、硬件型号、输入位模式、输出位模式、采样 domain 与反例；有限实验不能把未限定的恒等舍入或整个浮点 domain 证明为真。
