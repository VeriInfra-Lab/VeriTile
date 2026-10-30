# 浮点值模型的原语绑定实验

这些实验用于找出新逐算子/位值合同与具体编译器、设备之间的差异。它们不是原有 ADD-COMMUTE 等代数重写实验的替代品，也不会自动解除 Lean 前提或证明整个 GPU 核正确。

当前可执行的标量探针先由 Lean 的 `FP.ScalarOps` 生成精确参考，再执行独立 Triton 表达式作位比较。含 836 个加、减、乘、除、真正负号和绝对值案例，覆盖近舍入中点、次正规数、正负零、无穷、信号/安静 NaN。普通 `/` 与 `tl.div_rn` 分开报告；前者不能无证据地绑定到后者的模型。

```bash
python experiments/floating_point/value_validation/run_scalar.py --generate-reference /tmp/scalar-reference.json
python experiments/floating_point/value_validation/run_scalar.py --run /tmp/scalar-reference.json --output /tmp/scalar-native-run
```

需要仓库 Lean 环境在 PATH。第二步还需要 CUDA、PyTorch、Triton。目标文件/目录必须不存在，避免覆盖证据。保存 exact input/output words、当前软件源码哈希、GPU和编译版本、TTIR/TTGIR/LLVM/PTX及其哈希。NaN payload 和正负零位差异保留为 mismatch；不会使用数值近似相等掩盖它们。当前软件算术采用 nearest-even、不flush输入/结果、算术NaN规范化为正quiet NaN；负号单独翻转sign bit并保留payload，绝对值清除sign bit并保留payload。可匹配的硬件域和NaN策略必须据实际结果重新审查。

无缩放 FP4 的 [N21 逐位采集工具](fp4_unscaled/README.md) 已另行交付，使用原始两个 JIT 函数和 Lean 导出后独立核对的全部 256 字节参考。从仓库根目录运行：

```bash
python experiments/floating_point/value_validation/fp4_unscaled/run.py run \
  --reference experiments/floating_point/value_validation/fp4_unscaled/reference \
  --output /tmp/fp4-unscaled-native
```

采集配置包含非零指针偏移、尾部和额外完全 masked program；按原始 bf16 位比较活动输出，并检查整个分配范围。物理别名和内部操作数尚未覆盖。编译与源码绑定状态、数值匹配和 GPU 执行结果分开记录。此工具不代表完整原生一致性验证已完成。

逐任务还需检查以下关系，不能用这组标量样本代替：

| 待绑定关系 | 域/配置及应保留的信息 |
|---|---|
| 原始载入/存储位保持 | 任务源码实际声明的fp16/bf16/fp32/fp8等存储格式（抽象.real不推断为fp64）；NaN payload、-0、非零地址、masked tails；普通copy与数值cast分开 |
| 标量运算与模型一致 | 每个实际可达算子的输入位、操作顺序和目标精度；固定rounding/FTZ/DAZ，确认未把独立mul+add融合 |
| 存储格式转换 | 实际source/destination dtype、舍入方式、overflow/subnormal/NaN；fp16→fp32编码与回存需分别核验 |
| 常量/索引转浮点 | 原始literal、fromNat/fromInt等转换站点；记录目标精度和位编码，防止常量折叠、不同舍入或隐式类型提升改变模型输入 |
| 浮点比较 | 实际lt/le/eq条件、NaN、±0及边界值；保留比较结果与由其选择的分支，不能仅检查最终近似输出 |
| maximum/argmax | 非NaN域；±Inf与±0；最小索引tie；max结果来自哪个输入位模式；stage2的原始索引payload |
| reduction | 提取原核真实归约树/分组/累加精度；逐步结果与相同树的软件oracle比较。Delta等顺序软件fold不能直接当作原GPU树 |
| exp/log/pow/sin/cos/rsqrt | 精确指令/库版本、输入域和实际位输出或明确误差关系。后台回调不是Real精确函数，也不预先假设代数恒等式 |
| 浮点转整数 | 源精度、目标signed/unsigned width、toward-zero或其他实际模式、有限可表示域及域外行为；与llrint分开 |
| 整数点积和位域 | 保留int8/int32/packed word的真实signedness与宽度；算术右移及shift合法范围；int8点积的32位wrap/饱和模式和累加站点，不将数学Int之和视为无溢出的硬件结果 |
| 原核组合 | 编译原始任务及真实shape/stride/constexpr/fusion参数；确认上述原语在实际站点按同一方式落地，并保持mask/alias/索引范围 |

有限测试只能支持已记录配置和输入上的经验判断、或给出反例。未实施的项目保留待验证状态，不能因没有额外代数规律而记为“无需实验”。
