# 剩余浮点值模型修复：进行中

最新状态：全部新增源码修复包已准备并经根复核，正在统一构建及验收入口集成。当前 [证据索引](delivery-index.json) 收录 48 个包，[硬件实验清单](native-experiment-tasks.md) 为 37 项，均未执行 GPU 实验。统一构建按冻结阶段衔接，最终计划为 3,783 个模块；最终全部通过前不发布完成结论。下文保留各阶段记录，其中的“当前”与数量属于当时快照。

上一轮原 41 项的冻结验收记录仍见 [旧验收包](../acceptance-after-value-repair/receipt.json)。本目录记录随后开展的工作；它尚未发布新的 184 项总统计，也不将增加 `FPCorrect.lean` 文件自动当作任务完成。

本轮已补的范围包括 `quantize_copy_kv`、三个 token attention 任务、两个 Flash 任务、`matmul_tma` 和 `lora_expand_gemv`，并修正已有 `rowwise_quantization_triton` 的 maxnum 模型。两个 LayerNorm、layer_norm_ops、matmul_triton1/2 与卷积也已完成阶段冻结；共享 maximum/FP4 前端已通过定向合并检查，后续任务继续并行。每项的具体范围与存储格式单列；旧的全核 lowering 结论不作为全循环浮点值证明。

## 已形成的 Flash 阶段结果

[定向 Lean 编译记录](flash-stages/receipt.json)记录两个 Flash 任务的 10 个新模块，15 条公开记录。其中 14 条是参数化执行到输出/内存保持合同，另 1 条是 Phi 旧范围中的最大值递推恒等式，后者不能算执行证明。每条公开记录均包含 `#axiomsClean` 检查，另有 12 项参考定义非循环性检查通过；[独立语义复核](maxnum-profile-audit/receipt.json)已完成；正式总表集成仍待完成。

- LLaMA：归一化、最终拷贝、显式最大值图各有后端合同及软件 binary32 合同。归一化和拷贝保留原阶段的精确算法投影。最大值图显式使用 `max2`，没有声称旧前端的比较选择图已被修好；本任务因此保持未整体升级。
- Phi：masked 归一化、拷贝、accumulator 更新、sum-exp 更新均有完整阶段合同，四个源阶段的算法投影已检查。基本加减乘除使用具体 binary32 运算；两项更新属于保留算子参数的合同，不能算完全闭合的位合同：`exp32` 是明确的外部算子参数，没有替换成实数指数函数。
- 这些 Flash 合同的内存标签采用擦除后的 `.real`，但载荷是 `FP.Value .fp32`，运算调用实际 `FP.div/add/sub/mul`，不是数学实数运算。与实际 typed storage、分配和 launch 的对应仍需独立绑定。两个 attention/量化分支正在采用保留 `.fp32` 存储标签的更强模型。

## 完成定向检查的量化结果

`quantize_copy_kv` 的 7 个模块、22 项公理检查通过，详见 [编译与范围记录](quantize-copy-kv/receipt.json)及[根代理复核](quantize-copy-kv/root-review.json)。主合同覆盖所有位输入与任意未定义转换字节策略，保持完整原 head mask、全部 BLOCK_DMODEL 列及双输出；active-defined 的加强结论另给 toward-zero 与策略独立性，不把它作为主合同的新输入前提。新归约已采用显式 maxnum，native 对应仍待验证。

## 已知不匹配：必须修复，不能当作未知实验项

本地 Triton 3.8.0 的 `tl.maximum` 默认 `PropagateNan.NONE`，落到 `maxnumf`；源码证据见 [绑定的源码片段与哈希](triton-maximum-source.json)。旧 DSL 将它展开为比较后选择。输入为 `(1, NaN)` 时，两者结果不同；`FlashDecode2Llama.FPStages.compare_select_is_not_maximum` 已通过 Lean 内核检查。没有通过新增非 NaN 前提避开这个差异。

新 LLaMA 最大值模型改用显式最大值原语，并为单 NaN、双 NaN、正负零写出软件策略。共享前端已在独立构建目录加入真正的浮点 maximum 节点，NaN 双向、嵌套、广播及精度回归已通过；兼容模块和旧任务迁移仍在进行，尚未合入当前工作区，因此原核的最大值行为尚未据此验收。

量化归约也在检查同一类问题：排序 rank 的 NaN 选择不能自动代替默认 `tl.max` 的 maxnum 归约。`quantize_copy_kv` 已切换到独立 maxnum 模型并通过定向检查；已有 `rowwise_quantization_triton` 的无额外数值域合同已按 maxnum 修正，6 个模块及 12 项公理检查通过；公开命题文本不变，但定义的语义已修改，详见[新修订记录](rowwise-maxnum/receipt.json)和[根代理复核](rowwise-maxnum/root-review.json)。旧冻结包不能替代本次修订记录。其他含 `InputDefined` 或非 NaN 前提的任务须先核对反例是否落入原域，不能仅因导入相同 helper 就宣称反例成立。

## 仍需验证的关系

实际 `/` 指令与软件除法（包括普通除法是否等同单次正确舍入）、FTZ/DAZ、NaN/正负零策略、exp 指令、融合、归约树、类型/地址/存储和 launch 对应分别验证。存在已知反例的关系优先修改实现/模型；未知关系保留明确的实验任务。当前没有 GPU 执行结果，也未重跑本轮八库总构建。

已对两个 Flash 原 Python kernel 做 [4 次离线编译](flash-native-lowering/receipt.json)：AST 不变、fp32、SM90、BLOCK_SEQ=8/BLOCK_DMODEL=32，分别启用/禁用融合。TTIR 明确包含 `arith.maxnumf`；PTX 使用 `max.f32`、`ex2.approx.f32` 和 `div.full.f32`，默认融合还出现 `fma.rn.f32`。因此不能把分步舍入、精确指数或正确舍入除法的软件图自动当成默认编译产物。禁用融合配置仅消除已观察的 FMA；其他对应关系仍待验证。这是编译证据，不是 GPU 运行或数值通过证据。

## 新完成的完整循环与矩阵块

- [Q/K score 记录](token-attn-llama2/receipt.json)：10 模块、26 项公理检查。任意原 scale 的后端合同、显式 host encoder 的软件合同、任意有理 scale 的具体舍入特化分别列出；后两者不混为任意 host 标量 ABI 已闭合。[根代理复核](token-attn-llama2/root-review.json)无源漂移。
- [两个 PV 循环记录](pv-contracts/receipt.json)：16 模块、34 项公理检查，保留任意循环次数、全部尾部乘法与 Mistral 独立 mask。具体软件实例采用显式逐 lane 求和，GPU 归约树待绑定。[根代理复核](pv-contracts/root-review.json)无源漂移。
- [matmul_tma 记录](matmul-tma/receipt.json)：6 模块、11 项公理检查、2 项参考非循环检查。两个原输出分支均有完整执行/输出/frame 以及精确算法投影；顺序软件 dot 是明确实例，不能自动替代 WGMMA。另有[两个原核离线编译结果](matmul-tma-native/receipt.json)，未运行 GPU。
- [LoRA expand 记录](lora-expand-gemv/receipt.json)：8 模块、18 项公理检查，完整任意 N 输出循环、4 种 half/bfloat16 输入组合 × 3 种浮点输出格式，允许 input/output 别名，保留 K padding 运算。其他输出格式没有覆盖，原 ADD_INPUTS/CAST_TYPE 分支限制保持。该记录也纠正旧规划中“sum 自动升 fp32”的错误；当前软件实例按实际乘积类型执行 half/bfloat16 加法。

Q/K、两个 PV 与 rowwise 的[额外 8 项参考非循环检查](reviewed-reference-audit.json)通过；这验证独立结果定义不引用内核执行本身，不代替数值或硬件证明。

本轮各任务/修订的待验硬件关系与具体输入场景见[实验清单](native-experiment-tasks.md)。其中已有反例的前端/数值模型差异需要修正，不能把它们混入“等待实验验证为真”的假设。

`matmul_tma` 的[独立代理复核](matmul-tma-independent-review/review.json)已通过：6 模块源码/日志、164 个输入及原核离线编译产物哈希一致；软件范围内未发现阻断，硬件对应仍待完成。

`matmul_triton1` 的[完整循环记录](matmul-triton1/receipt.json)覆盖任意 K=BK×steps、全部原输出与别名，8 模块、16 项公理检查通过。真实 fp32 点积与独立外层加法均保留，`allow_tf32=False` 没有丢失；任意有效软件归约树的证明没有被算成 native dot 已绑定。[根代理复核](matmul-triton1/root-review.json)验证源码、依赖和日志无漂移。

两个 LayerNorm 的[19 模块记录](layernorm-contracts/receipt.json)与[根代理复核](layernorm-contracts/root-review.json)已完成。6 个新公开规格、51 项公理检查通过，5 个旧公开命题和原 Lean/Python 文件不变。Liger 的填充方差项、stream 的完整三遍循环与未定义 W 尾部运算均保留。sqrt/rsqrt 与非平凡常量转换仍是显式绑定，因此这两项不会被计为闭合的位解释器或 GPU 正确性。

LoRA 的[根代理复核](lora-expand-gemv/root-review.json)也已完成；本次 LoRA、matmul_triton1 和两个 LayerNorm 的[额外 8 项独立参考非循环检查](loop-reference-audit.json)全部通过。

FP4 两个缺失任务的上游原文件已[按固定 revision 取回](fp4-upstream/receipt.json)。全部 16 种编码的位转换辅助证明与原未缩放源码解析已通过；[前端冻结包](fp4-frontend/receipt.json)有 31 个模块的独立定向重编，包含完整读写、负零/NaN 载荷和整数转换回归。完整未缩放 kernel 已完成下述阶段冻结；带缩放版本继续处理中。interleave/runtime bitcast 修改尚未合入共享源码，正式总表尚未更新。

`matmul_triton2` 的[8 模块记录](matmul-triton2/receipt.json)已冻结：23 项公理检查、3 个公开规格，完整 K 循环、group、六 stride、别名与最终 mask 保留。三个软件输入模式（完整 fp32、TF32 近偶、TF32 近离零）明确分开，未推断 native 默认模式。原 A/B 只有 K mask，因此部分 M/N tile 会读逻辑 shape 以外地址；这是源级分配/地址问题，不是可用数值实验验证为真的假设。

`layer_norm_ops` 的[9 模块记录](layer-norm-ops/receipt.json)已冻结：42 项公理检查、9 个公开规格覆盖原 8 个阶段，保留独立预计算输入和单行限制。新单行源码恢复 DW/DB 的 +0 累加，负零实际执行见证通过；没有声称它与省略该操作的旧数学 helper 浮点等价。

FP4 未缩放原 helper/kernel 的[两个原 AST 离线编译](fp4-native/receipt.json)通过，块大小 16 与原 host 的 512。TTIR 确认高 nibble 在前、uint8 减加回绕、int32→fp32 bitcast 与 bf16 转换；这仍不是 GPU 运行或完整任务验收。

最大值共享前端及 LLaMA 的[原语补丁](maximum-core/receipt.json)、[任务迁移](maximum-flash-decode2-llama/receipt.json)均已在私有目录通过并完成根代理包复核，尚未合入工作区。LLaMA 已有真实 parsed maximum 的完整阶段执行与 NaN 非空见证，删除了先前适配图的多余 cast。

`int8_matmul_quantization` 的[第二版迁移记录](maximum-int8-matmul-quantization-v2/receipt.json)已补齐原输入域证明并通过[根代理复核](maximum-int8-matmul-quantization-v2/root-review.json)：84 个模块、100 项公理检查通过。五个旧域定义逐项绑定原源码；旧域推出新域，且 running maximum、scale、quotient 均相等，不新增非 NaN 前提。第一版记录保留；新版本仍等待统一集成。

新增 matmul_triton2 与 layer_norm_ops 的根代理复核及[额外 6 项独立参考检查](additional-reference-audit.json)通过；所有本轮记录仍等待统一集成与总构建。

`triton_conv2d_fwd` 的[完整三层循环记录](conv2d-contracts/receipt.json)与[根代理复核](conv2d-contracts/root-review.json)已完成：10 模块、30 项公理检查、3 个公开规格及 [3 项独立参考检查](conv-reference-audit.json)。完整软件合同使用真实 half 舍入，不需要旧 `half=id`；主合同保留任意 undef。零填充是单独的可选推论，实际执行已证明不同 undef 可产生不同输出及 `0×Inf→NaN`。原 fp16=true 范围保留；native dot/融合/undef 语义仍未绑定。

`mixed_sparse_attention` 的[最大值迁移记录](maximum-mixed-sparse-attention/receipt.json)与[根代理复核](maximum-mixed-sparse-attention/root-review.json)通过：79 个模块、56 项公理检查，执行图直接等于 parser 结果，删除局部 `restoredLoop` 替换；原独立 reference 和 5 个公开规格字节不变。完整 forward 仍是保留 dot/exp2/cast 等回调的合同。

`matmul_triton2` 的输入越界已形成[静态地址反例](matmul-triton2-address-boundary.json)：M=N=16、K=32、原 32×64×32 tile 下，两个 K-mask 有效输入地址均为 512，恰好越过各自的 512 元素连续分配。K 属于原接受范围；输出 mask 不能保护前面的输入读取。这是 Lean 检查的地址结论，未声称 GPU 实际发生故障。

共享 maximum 的[影响清单](maximum-impact-audit/impact.md)记录 29 项源码使用者；已迁移者与待重编/适配者分开。maximum 与 FP4 两套前端补丁在第三个私有目录无冲突合并，[合并检查包](combined-frontend-consumers/receipt.json)及[根代理复核](combined-frontend-consumers/root-review.json)已冻结：84 个核心/回归模块和 233 个下游模块全部通过，103 项公理检查通过。165 个旧声明中 164 个逐字不变，仅 1 个 parser 形状辅助命题改为实际 maximum 节点；旧数值合同未增加前提。这是 softmax_flaggems、kldiv_ops、chunked_cumsum_fwd 和 AttentionForwardClosedForm 及依赖的定向通过，尚未发布全库兼容结论。正式验收统计保持原冻结版本。

`triton_matmul` 的[两格式完整循环记录](triton-matmul/receipt.json)和[根代理复核](triton-matmul/root-review.json)通过：10 模块、22 项公理检查、4 个公开规格；原 4 条声明和 Lean/Python 源码不变。[4 项独立参考检查](triton-matmul-reference-audit.json)也通过。half/E4M3 均保留三参数 dot 的 live accumulator，闭合软件实例采用逐 lane fp32 FMA；E4M3 有全部 254 个有限编码的往返检查。

原 `triton_matmul` 两种 host 配置的[离线编译](triton-matmul-native/receipt.json)通过；实际指令为 WGMMA，E4M3 输出含 `cvt.rn.satfinite.e4m3x2.f32`。这不建立 WGMMA 与顺序 FMA 的等价，也没有 GPU 数值结果。FP4 的 Nat→int32 保宽修改另有 [9 项候选影响清单](fp4-int32-impact.json)，已完成[逐项审计](int32-consumer-audit/receipt.json)和[根代理复核](int32-consumer-audit/root-review.json)：3 个 packed-int4 任务修复并通过全部 34 模块；445 个旧声明头、5 个 DSL 声明不变，没有新增 int32 输入上界。4 个当前相关 cast 子图未变，其中 block_sparse 的独立旧 LegacySummary 另有真实不兼容。2 个 retention 的无界旧数学域有 signed32 截断反例；其 6 个相对 FP 模块通过不等于绝对输出值证明。语义反例与旧 Source 的递归深度编译问题分别记录，未当作全库兼容通过。

`matmul_triton_autotune` 的[完整循环记录](matmul-autotune/receipt.json)和[根代理复核](matmul-autotune/root-review.json)通过：9 模块、18 项公理检查、2 个公开规格，另有 [4 项独立参考检查](autotune-reference-audit.json)。两 activation 分支、原 modulo/pid 映射、live dotAcc 与 half 输出均保留。五种浮点 storage 有后端合同，原 fp16 测试配置有闭合的软件位合同；软件顺序 FMA 不自动等同 native tensor-core，autotune/launch 仍需绑定。

`softmax_reducev` 的[完整在线循环记录](softmax-reducev/receipt.json)和[根代理复核](softmax-reducev/root-review.json)通过：11 个任务模块、2 个公开规格，104 个旧声明文本不变；包内另有 65 个依赖模块及总计 60 项公理检查。初态、完整循环、signed gather、最终除法、全部输出和 frame 已贯通，[4 项独立参考检查](softmax-reducev-reference-audit.json)通过。具体 fp32 标量/casts 已闭合，exp32 和明确选定的归约树仍是参数，因此分类保持后端值合同。该补丁依赖新 maximum 前端，仍在私有目录等待统一集成。

`fp4_to_bf16` 的[完整未缩放内核记录](fp4-unscaled/receipt.json)与[根代理复核](fp4-unscaled/root-review.json)通过：11 个任务模块及一个源码桥接审计、37 项公理检查、160 个源输入无漂移，另有 [4 项独立参考检查](fp4-unscaled-reference-audit.json)。覆盖任意 count/block/pid，唯一输入条件是实际 active uint8 单元的位宽；masked undef 和未读单元不受限。完整 source 的输出等于初态表，全部 frame、callback 独立性和 undef 独立性已证。bf16 cell 载荷使用显式 raw16 容器；它不等于 native 字节别名或跨 program 调度证明。

`lightning_attention` 的[阶段值合同](lightning-attention/receipt.json)与[根代理复核](lightning-attention/root-review.json)通过：9 个任务模块、70 个含依赖模块、106 项公理检查；39 个旧声明和原文件字节不变。3 个 dot 阶段各有后端合同及具体软件合同，另有全部位值的 GradPre 拷贝合同；[8 项独立参考检查](lightning-reference-audit.json)通过。完整固定块 body 已证明，原外层 NUM_BLOCK/NUM_CBLOCK 循环仍只有 lowering 结论，不能据此升级为全 launch 数值正确性。

带缩放 FP4 暴露的显式 int16 丢失问题已有[独立前端修复](int16-frontend/receipt.json)及[根代理复核](int16-frontend/root-review.json)：5 个核心模块、15 个回归模块、89 项公理检查通过。任意 Int/Nat 的执行等于低 16 位有符号结果，FP4 两个转换节点逐个保留；原 int32/无符号/浮点转换和四种 multinomial 源 profile 回归通过。该修复只覆盖显式转换，不声称所有整数算术都已有机器位宽语义，仍等待统一集成。

`matmul_kernel` 的[完整循环包](matmul-kernel/receipt.json)与[根代理复核](matmul-kernel/root-review.json)通过：9 个新模块、原 main 及独立旧规格审计共 11 模块、19 项公理检查；原 2 个公开声明及文件字节不变，另有 4 项独立参考非循环检查。完整任意 BM/BN/BK/T、二维 pid、输入 %4096、全部无 mask 读写和零次循环保留，仅用原 BN≤4096。具体 half→fp32 live FMA→half 位模型已闭合；原 host K4096 的实际 trip、地址、tensorcore 舍入/顺序仍另行绑定。

`sgmv_expand_slice` 的[双图完整值证明](sgmv-expand-slice/receipt.json)和[独立复核](sgmv-expand-slice/root-review.json)通过：17 个本轮模块及 61 个同核心依赖记录、86 项公理检查；47 个旧声明和 2 个旧公开规格逐字不变。原 Python 的 singleton tuple 使两种输入配置编译失败，仅去掉逗号的候选两种均离线编译通过。源 dot+add 与实际候选 dotAcc 分别证明；软件两图已有位差反例，不能采用普遍融合等价。7 项参考检查排除了执行器循环定义。

带缩放 `fp4_to_bf16_conversion` 的[候选完整证明](fp4-scaled/receipt.json)和[独立复核](fp4-scaled/root-review.json)通过：14 个模块、50 项公理检查、172 个输入源及 34 个离线工件哈希核对。原 pow(fp32,int16) 在 5 个配置均编译失败，唯一指数转 int32 的候选均离线通过。主合同保留任意长度尾部，仅要求活动 packed 输入为 uint8；原 floor scale mask 在 count17 的活动输出32 依赖 undef，已给出完整执行反例。全部256个 scale 编码及 bf16 舍入/乘法已有软件位证明，native powif 和硬件行为待实验绑定。这份冻结包明确保留旧 parser 的单处 int16 AST 适配，后续统一前端桥接另行复核。

`matmul_leakyrelu` 的[完整双图证明](matmul-leakyrelu/receipt.json)与[根代理复核](matmul-leakyrelu/root-review.json)通过：19 个模块、46 项公理检查、8 项独立参考检查；原 2 个公开规格及原 Lean/Python 字节不变。完整 cdiv K-loop、K/M/N 尾部、grouped 双取模及两个 activation 分支均保留。四个 SM90 离线配置均观察到 live dotAcc 和 activation 的 x*slope，故给源图/编译候选各自的 backend 与 half 位合同。三轮完整有限 half 执行已证明软件输出 +0 与 -1 的差异；未把该软件反例称作 GPU 实验结果。

`matmul_leakyrelu_fp8` 的[完整循环证明](matmul-leakyrelu-fp8/receipt.json)与[独立复核](matmul-leakyrelu-fp8/root-review.json)通过：19 个模块、50 项公理检查、8 项独立参考检查，119 个交付工件哈希核对。原 2 个公开规格及 Lean/Python 源码不变；闭合软件合同同时覆盖 half 与 E4M3FN 输入、固定 half 输出。原真实测试使用 half，名称不被当作 dtype 证据。8 个原 autotune 配置的 dtype/activation/fusion 组合离线编译通过；原 slope*x 与编译 x*slope 图分开，hardware MMA 行为仍待验证。


### attention_kernel 最大值投影兼容复核

`maximum-attention-kernel-compat/` 冻结两个完整实数投影模块的兼容修复：109 个旧声明头逐字保留，完整编译和模块公理审计通过。这是数学投影与新最大值节点的兼容检查，不新增浮点或硬件完成项。回执的预算未增加字段仅相对交接版本；相对最早原源新增的全局预算已在 root-review.json 明确注明。正式总表未更新。


## 统一 maximum 使用者复核与新增完整合同

[maximum-all-consumers](maximum-all-consumers/receipt.json) 覆盖影响清单的 29 个任务及传递依赖，最终 **1,083/1,083 模块通过**。45 个兼容修复文件中的 1,086 个旧定理/规格声明头逐字保留，另新增一个实数最大值桥接模块。源码、日志及 1,129 个导入/目标对象哈希均核对；初次失败与修复日志单独保留。四个已退役局部模型明确固定为历史图，其中 block_sparse 的旧 Nat 概要不再冒称实际 signed32 源码。这是前端兼容闭包，尚不是全八库集成或硬件验收。

[triton-linear-activation](triton-linear-activation/receipt.json) 已通过根代理复核：15 个最终模块、76 个含核心记录、204 个依赖输入，旧 68 个定理/规格头保持不变，另有 7 项独立参考检查。两份完整后端合同覆盖全部激活字符串和 bias/save 两个标志；仅附加 ReLU/identity 特化闭合软件位运算。tanh/erf/division/dot 仍明确保留回调。原 fast_gelu 三个配置编译失败，唯一 constexpr 常量候选三个通过；不得把候选或固定零输入见证混入原核全域正确性。

[multinomial-sampling](multinomial-sampling/receipt.json) 已通过根代理复核：12 个模块、29 项公理检查、74 个输入和 40 个离线工件哈希核对，另有 6 项独立参考检查。原 11 条前置语句、全部 12 条循环体和最终 masked store 已贯通，四种 score 存储格式均有完整后端/软件位合同。软件保留显式扫描/最大值树和真实 Philox；两轮 alias/尾部执行证明第一次选41后被99覆盖。它描述源码舍入行为，不证明理想概率分布；实际树、FTZ、存储/地址与GPU绑定仍待实验。

本轮目前列出 [29 个任务/修订的硬件对应实验](native-experiment-tasks.json)，每项绑定确切证明回执和输入场景。冻结交付包见 [统一索引](delivery-index.json)；包数不等于完成任务数，正式184项统计仍待集成重算。

## IV matmul、scaled FP4 直接桥与遗漏审阅绑定

- `iv-dependent-matmul` 已完成 root 复核：71 条编译记录、196 个依赖输入、53 条旧定理/规格及旧主文件字节不变；完整旧 `pre_load` 范围有源码/编译图两套 backend 合同与两套软件位值合同，6 项独立参考审计通过。TF32 MMA 硬件对应仍待验证。
- `fp4-scaled-int16-bridge` 已完成 root 复核：23 个模块、172 个输入、84 项公理检查；完整47语句直接采用修复 int16 前端后的解析结果，5 项独立参考审计通过。旧 public/reference/witness 字节不变；依然仅适用于指数显式转 int32 的 Python 候选，原5项离线编译失败保留。
- `flash-stages` 与 `matmul-tma` 的既有 receipt、源码、对象、日志和离线工件重新核对，补入显式 `root-review.json`、输入快照及独立参考审计。Flash 仍是原接受的阶段范围；TMA 保留完整单 tile 范围。
- delivery-index 现绑定36个包/修订（不是36个新完成任务）；native-experiment-tasks 现有30个任务/修订，均无 GPU 执行结果。
- 私有统一集成计划为3684个模块；IV/scaled FP4 的21个新增模块已复制为待补充构建根。构建脚本修复了将注释内 Python import 误识为 Lean 依赖的问题；原失败日志与263条已通过记录另存，随后继续精确哈希校验构建。全库统计/验收尚未更新。

## Spinning lock 完整任务局部源码桥

`spinning-lock-source` root复核通过：166工件、11模块、286依赖对象、5条公开规格；其中新增2条完整源码合同，旧3条SC模型合同字节不变。独立重放生成的26语句typed AST与6项原Python调度测试逐字节一致，6项控制/参考非循环审计通过。最终参考从配置、初始锁和调度生成事件，再从初始内存重算所有值，已证明控制轨迹不依赖浮点后端或P/C值。

这仍是明确SC、逐lane序列化的源码模型：共享DSL admission、GPU内存序/竞争自由与终止性没有完成。原锁复用缺少happens-before的反例保留；publish-once候选只离线编译，不算已修复。现index37包，实验清单31项，统一构建待补充根32个。

### Aligned 全循环值合同与独立统计重放

`attention-aligned-value`：完整原 fp16 范围的 12 个模块、两条参数化值合同已独立复核，7 项参考依赖检查通过。保留真实 half 舍入，去掉全域 half=id；host scale、exp2、typed dot 仍是显式原语接口。原编译图融合 QK/PV dot，单列为硬件/编译图待对应，不能计作全软件或 GPU 正确性。

`value-metadata-candidate`：36 包/30 实验任务快照在独立空目录重放，816 个输入、747 条历史声明、94 条新增声明和 18 项拒错测试通过。该快照不含后续 spinning/aligned，不改变正式统计；下一版需连同统一构建和 Lake 注册重新绑定。当前交付索引共 39 个证据包，硬件实验清单 32 项；包数不是完成任务数。

### Retention signed32 参考公式修复

`retention-wrapped-indices`：两个任务的完整前向、输出、后向状态和 Q/K/V 梯度范围已重新检查，43/43 模块通过，9 项独立参考依赖检查通过。7 个旧证明文件修复、4 个新辅助/反例文件，24 条公开 specification 和两个 kernel 源文件保持原文；参考定义明确改为实际 signed-low32 索引，不增加 BT 上界。旧自然数索引公式的因果 mask 错误有 Lean 反例，普通范围下的等价单列为推论。

这关闭的是源码/参考索引不一致；浮点 exp2/dot、所有其余机器整数运算和真实 GPU 行为仍保留各自的对应义务。当前统一构建仍使用原冻结输入，已登记下一轮应用本补丁；不能用这份独立检查直接改写正式验收。索引目前 40 个证据包。

### 统一统计候选第二版

[value-metadata-candidate-v2](value-metadata-candidate-v2/receipt.json) 已独立重放：865 个输入、25 项拒错测试通过，747 条历史公开声明和旧 41 项注册表保持不变。当前覆盖 29 项扩展、99 条新增参数化值合同；另外 1 条递推恒等式和 1 条内存 frame 单列。共享 DSL 与任务局部 SC 源码桥分开统计，源码、编译图和修复候选不能合并充数。

当前交付索引为 41 个证据包，原生实验清单为 34 项且均未执行 GPU 实验。本候选包含 Retention 参考修订，尚待统一编译与验收入口集成；正式历史验收记录保持原样。

### LayerNorm 与 StreamK 完整源码合同

`layernorm-lock-source` 根复核通过：三个完整内核、16 个模块、11 条公开合同，54 项公理检查；8 项原 Python AST 测试与10项前端变更测试独立重放，10 项参考非循环检查通过。新增5条源码值合同和1条SC互斥合同；旧4条模型值合同和1条frame保留。前向 sqrt 仍是原语接口；后向及最终归约有完整软件位模型。

`streamk-source` 根复核通过：五个原 JIT、64条语句、9个模块和7条参数化源码合同；7项原 Python AST 测试、12项host公式检查独立重放，7项参考非循环检查通过。两波均计算完成才产生完整host结果；原源码越界读、half舍入损失、原子顺序差异和阻塞反例保留。

两项采用明确的任务局部 SC 语义，未冒称共享 DSL、GPU内存模型或无条件终止性。索引现43个包，原生实验清单36项；统一编译与新的验收入口仍在集成。

### isfinite 与新 bitcast 语义兼容

`isfinite-bitcast-compat` 根复核通过：45/45模块、278项公理检查，143个源码输入及98个复用依赖对象均绑定。新版求值器开始支持整数到fp32 bitcast，因此内部通用重定位引理改为使用子表达式重定位结论；该辅助引理没有外部调用，6条公开规格及Kernels/Correct/FPEquiv完整字节不变，没有新增公开输入或数值假设。补丁已排入后续统一构建，索引现44个包；硬件实验任务仍为36项。

### Triton attention 完整三阶段

[triton-attention-value](triton-attention-value/receipt.json) 完成 26 个新模块及独立审计，74 项公理检查通过；根复核另检查了 9 个独立参考定义。完整预处理、前向循环、后向双层循环的值、读回与 frame 均有参数化证明，原四条公开声明和旧主文件保持不变。六条新规格中，五条保留数值原语回调，一条完整预处理在 fp16/bf16/fp32 存储下闭合为软件浮点结论；三个固定成功见证不计入参数化证明数。

原核 18 次离线编译为 10 次通过、8 次 dtype 不匹配失败；默认 FMA、dotAcc、exp/div 指令及 host scale ABI 的边界都在实验清单中明确列出。新的三阶段合同均不使用 half=id。

### 统一构建覆盖与记录校验

[integration-build-audit](integration-build-audit/receipt.json) 独立核对首轮八库的 3,728 个注册模块及全部 3,297 个任务 Lean 文件，未发现漏项。构建器增加日志哈希复用校验和编译前后直接依赖对象一致性检查，已在隔离复现中验证，随最终阶段应用。另有 24 个 CI 明确要求的辅助 Lean 检查待最终源码稳定后补跑；没有把范围外辅助文件算作已通过。

### 184 项统计候选第三版

[value-metadata-candidate-v3](value-metadata-candidate-v3/receipt.json) 已在独立目录重放：932 个输入、33 项拒错测试全部通过。新增125条公开声明中，121条为参数化值合同（含6条独立SC模型），2条为frame、1条为递推恒等式、1条为协议合同。32项扩展的所有声明与各自源码、阶段及根复核记录逐一绑定。

按各项已接受范围，当前50项具备闭合的软件浮点模型、133项保留后端/数值原语参数、1项仅有修复候选。软件与后端结论都不自动代表GPU实现正确。第三版冻结的是46包/37实验任务的输入快照；当前索引增加本统计包后为47包。正式统一构建和主工作区合入仍待完成。

### 新的源码与验收入口

[current-source-adapters](current-source-adapters/receipt.json) 已完成独立安装复核，79项检查通过；11个安装文件有精确来源和目标映射。新入口登记原源码、候选、编译图与SC模型，并核对源码、输出对象、日志、直接导入依赖、Lean程序和隐式Init。80处受控源码迁移逐项绑定已复核包，最终安装前必须匹配确切after哈希。178个历史源码入口与23组原检查已经过中间回归；最终3783模块稳定后会重新绑定并执行。索引增加此包后为48包。
