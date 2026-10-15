# token_softmax_llama：完整来源复用桥与 exp 缺口实验

本包将 Llama 的原程序精确绑定到已冻结 Bloom 指数反例。**没有新增闭合指数算法，没有 task 分类提升，也没有新增公开 specification。** 两个新增 Lean 模块只是来源/完整旧合同/反例复用桥，不把 Bloom 的三个模块和 21 条检查重复计为新的数值结果。

## 原程序与完整旧范围

[source-ast-audit.json](source-ast-audit.json) 独立验证两任务 `_fwd_kernel_token_softmax` 和 `token_softmax_fwd` 的完整 decorated 段逐字节相同，且 AST 相同；整个 Python 文件不同，因为测试数据不同。Llama 默认测试使用 fp32、BLOCK_SIZE 8/16。父证据的 receipt SHA 是 `5a44de64505a9d9dd2df9def1d9484ca63de11e4c2df9a533696cffb1d73d08e`，本包保留 [父 receipt](bloom-parent-receipt.json) 与 [绑定说明](parent-binding.json)。

旧 `FPCorrect.token_softmax_typed_correctness` 本来就直接使用 Bloom 的完整 `TypedContract`/`TypedIO`，另以 Llama `TypedShape` 对应其原源。接受范围是全部 FloatDType 参数化存储、整行 load/−Inf padding→fp32 max/sub/exp/sum/div→目标转换→masked store/readback/frame。只保留原 width>0、输出 token stride≠0 条件。任意初态、地址/长度参数、alias、count=0、非有限值等旧域未缩小；`.real` 是抽象通道，不自动代表 native fp64。

[ActualSource.lean](bench/tritonbench_g/token_softmax_llama/Proofs/ActualSource.lean) 独立调用 parser 解析完整 typed 行核，证明到 Bloom typed kernel 及 Llama 旧 surface 的桥，并复用完整旧初态→输出/frame 合同。[ExpBindingGap.lean](bench/tritonbench_g/token_softmax_llama/Proofs/ExpBindingGap.lean) 明确复用已证的 fp32 正常输入和 fp16 次正规数可达构造。两模块、7 条命名 standard-axiom 检查实际通过；这是复用验证，不是 7 个新增数值闭合结果。

## 已证明的障碍与未证明事项

两个 fp32 指数操作数 `0xbf400006` / `0xbf400007` 经过实际常量 `0x3fb8aa3b` 的 RNE 乘法都得到 `0xbf8a7fb1`。父包已经用 `Real.exp_bound` 的精确有理 Taylor 界证明直接实数指数分别位于输出 `0x3ef1da02` / `0x3ef1da01` 的严格最近舍入中点区间。因此，任何确定性 ex2 函数都不能在两输入上同时等于直接 RN(exp)。

Llama 的默认 fp32 正常输入构造为 active row `[+0, x]`，32 lane 中其余均为 masked padding；没有用 NaN、无穷或次正规数制造该反例。参数化存储域也包含父包的 half 构造 `[0x3a00, 0x8006/0x8007]`，其真实 half 扩宽及 fp32 减法已证明；本包只复用，不另造假设。两可达证明的最大值树是明确的软件合法树，尚未绑定 GPU shuffle/smem 布局。

**这只否定内部 exp 阶段的全域直接正确舍入绑定，不证明最终存储 softmax 输出一定不同，不否定原任意后端合同。** 从指数到 sum/div/cast 仍可能有不同舍入结果；必须单独观察或证明。

## 本任务实际原核离线证据

[20 个 Llama 原核 profile](native/receipt.json) 全部离线编译成功：SM80/SM90 × fp32 × BLOCK_SIZE 8/16/32/2048/4096 × fusion false/true；warps 按原 host 分支选择 4/8/16。原 JIT 逐行提取，完整函数 AST 保持；保存实际 signature、metadata、IR/PTX/cubin 和整个 Triton、真实 ptxas/libdevice/Python 前后哈希。host 未显式指定的 `num_stages=3` 在采集配置中明确记录，不混称 host 源码参数。

[数据流](native-dataflow.json) 确认 76 条 sub→常量 mul→ex2.approx 的标量直线链。exp→div 的 12 条在可检查直线寄存器区间，其余 64 条跨同步/控制，明确 **NOT_PROVED**。除法仍是 `div.full.f32`；归约 maxnum/add，未证明完整 PTX 或 GPU 归约计划。22 个保守覆写/控制流拒错及 CPU 检查通过。

[与父包共有的 12 个 profile](parent-profile-comparison.json) 在保留所有可执行指令、寄存器、参数和控制标签，仅去除 debug 信息与空白后逐字节相同。这是文本证据，不是 native 数值证明。

[NVIDIA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2) 给出 ex2.approx 的近似语义；[div.full](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div) 也不是正确舍入除法。误差界不提供唯一位算法。当前没有另一个可证明与原图相同的纯 IEEE exp/div 实现；不能换名保留 callback 后升级软件分类。

## 可运行实验与真实状态

[experiment-contract.json](experiment-contract.json) 指定完整原核采集的分配、非零指针原点、grid、strides、mask tails、两组正常 fp32 输入和局部判定。默认 Llama runner 只跑 fp32；half 构造留作父包的独立证据。分别采集原 JIT 的完整原始输出/输入及 frame，以及另一个明确标记的指数指令诊断程序。

```sh
python verify.py
python replay.py --output /tmp/llama-fresh-proof-replay
/opt/conda/bin/python -B capture.py compile --output /tmp/llama-fresh-offline --arch 80 --arch 90
/opt/conda/bin/python -B capture.py run --output /tmp/llama-fresh-native --device cuda:0
```

当前[诊断离线](diagnostic-offline-final/report.json)两项通过；[实际 native 尝试](native-attempt-final/report.json)为 **NOT_RUN**、0 次 launch（CUDA 不可用）。没有完整 softmax expected evaluator：运行最多得到 CHARACTERIZATION，不能得到全核数值 PASS。frame 或已证明乘法局部不变量失败会单列并返回非零。设备/driver 对应、编译工具绑定与数值观察分别记录。

剩余事项是原 ex2.approx/div.full 的具体机器位实现或设备绑定、实际 max/sum 树以及全部接受 dtype 的转换/特殊值/地址对应。它们不因已知反例、来源复用或固定实例采集而闭合。所有主工作区、注册表、原声明和冻结父包均未修改。
