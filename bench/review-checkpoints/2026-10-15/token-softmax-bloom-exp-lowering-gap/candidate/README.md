# token_softmax_bloom：指数 lowering 缺口的正式反例与实验交接

本包没有把任务从 backend 升为 software，没有新增公开 specification，也没有修改主工作区、旧证明或注册表。它证明一个必须排除的补齐方案：原编译图的 `mul32(x, 0x3fb8aa3b) → ex2.approx` 不可能对所有合法指数操作数等于直接正确舍入的 `exp(x)`。这是指数内部阶段的反例；**没有证明最终 softmax 存储值必然不同**。

## 原接受范围保持

旧 `FPCorrect.token_softmax_typed_correctness` 覆盖整行初态加载/负无穷补位、fp32 max/sub/exp/sum/div、目标格式转换、masked store、读回与完整 frame。只有原 `width>0` 和输出 token stride 非零条件。存储参数是全部 FloatDType；其中五个具名机器格式与抽象 `.real` 通道应分开，`.real` 不是 native fp64 证据。没有新增有限值、正序列长度、分母非零、无 alias 或 Nat 地址上界条件。原源及声明快照/哈希见 [original-scope.json](original-scope.json)。

## 实际形式证明

三个新模块最终拓扑编译通过，21 条命名 axiom 检查均仅使用标准基底，见 [final-build.json](final-build.json)、[theorems.json](theorems.json)。它们是普通定理，不计入公共规格数量。

- `ExpLoweringGap.multiply_collision`：`0xbf400006` 与 `0xbf400007` 经具体 fp32 RNE 乘法均得到 `0xbf8a7fb1`。
- `real_exp_A_rounding_cell` / `real_exp_B_rounding_cell`：利用 Mathlib `Real.exp_bound` 的 16 项有理 Taylor 误差界，分别严格落入相邻 fp32 输出 `0x3ef1da02` 和 `0x3ef1da01` 的中点开区间。`output_cell_decoding` 给出四个邻接 word 的精确有理值；不依赖 CPU exp、GPU 或输出相等假设。
- `no_ex2_function_matches_both`：任何确定性 `Word → Word` SFU 都不能把同一乘后 word 同时映成这两个不同目标。这否定的是全域直接 RN(exp) 绑定，并不否定旧任意后端合同。
- `ExpGapReachable`：保留 fp32 正常有限数构造，行的两个 active 输入为 `0` 与上述负数之一，另外 30 lane 为补位。初态读取、最大值及减法具体落实到原完整数值 reference；未知 exp/rest 不影响这些指数前操作数。
- `HalfExpGapReachable`：原默认 fp16 存储也可到达同样操作数。两个 active half 输入是 `0x3a00`（0.75）与 `0x8006` / `0x8007`（负次正规数）。具体 half 编码扩宽、fp32 减法分别产生两个碰撞输入；证明完整 32 lane padding/max/reference 桥。不是把 `-0.750000…` 误当成可直接存储的 half 数。

两条可达桥采用明确的合法、带 seed 最大值树。它们不声称该树是 GPU 实际 shuffle/smem 计划。两个例子的构造是反例证据，不是用固定实例替代原全范围正确性。半精度转换和 sub32 已有软件位运算证明；硬件 FTZ/转换行为尚未观察。fp32 正常输入例用于区分该问题与 half 次正规数/FTZ。

## 原核编译与数据流

[原核 36 个离线 profile](native/receipt.json) 全部编译成功：SM80/SM90 × fp16/bf16/fp32 × BLOCK_SIZE 32/2048/4096（原 host 对应 warps 4/8/16）× fusion false/true。原 JIT 逐行提取且完整函数 AST 相等，未修改内核。实际 signature、constexpr、metadata、TTIR/TTGIR/LLVM/PTX/cubin、整个 Triton 包、真实 ptxas/libdevice/Python 前后哈希均保留。完整 host 分支审核见 [host-profile-audit.json](host-profile-audit.json)。这是三个存储 profile 的离线证据，不覆盖旧合同其余 dtype 或任意布局配置的 native 执行。

[数据流检查](native-dataflow.json) 在 36 个 profile 中确认 204 条实际 sub→常量乘法→ex2 链；fusion false 使用显式 `.rn`，fusion true 的部分文件保留默认舍入拼写。归约为 maxnum/add；除法是 `div.full.f32`。12 条 exp→div 关联处于检查器支持的直线寄存器区间，另 192 条跨控制或同步边界，明确标记 **NOT_PROVED**。本检查器不是通用 PTX 解析器或 CFG/内存语义证明；未知写、predicated/multi-dst 覆写和标量区间控制流会拒绝。22 个 CPU/拒错测试通过。

[NVIDIA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2) 将 `ex2.approx` 定义为近似指令；[div.full](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div) 也不是 IEEE 正确舍入除法。公开误差界不能确定唯一输出位值。不能用 `FP.div` 的 RNE 算法直接宣布这条 native 除法已经闭合。

## 可直接执行的最小实验

详细输入、指针原点、grid、原始位模式、frame 及判定见 [experiment-contract.json](experiment-contract.json)。执行前先运行 `python verify.py`。

```sh
/opt/conda/bin/python -B capture.py compile --output /tmp/bloom-fresh-offline --arch 80 --arch 90
/opt/conda/bin/python -B capture.py run --output /tmp/bloom-fresh-capture --device cuda:0
```

采集器明确区分两种程序：`exponent_probe.py` 是用于记录乘后/SFU 原始 word 的诊断程序；`native/original_jit.py` 是未修改原行核。原核会分别运行 fp32 正常输入与 fp16 次正规数两组，保存全部输入/输出物理分配、非零指针、mask tail、raw words、frame、实际编译 metadata/IR/PTX/cubin 和设备/driver 对应。诊断结果只对已证明的乘法碰撞及相同 SFU 输入进行局部检查；**没有完整 softmax expected evaluator，整个采集最多标 CHARACTERIZATION**。frame 或已证明局部不变量失败会单列并返回非零，不会被数值捕获掩盖。

最终两项诊断离线编译在 [diagnostic-offline-v3](diagnostic-offline-v3/report.json)；当前实际 native 尝试在 [native-attempt-v3](native-attempt-v3/report.json)，状态 **NOT_RUN**（CUDA 不可用，0 次 launch）。其他 diagnostic/native-attempt 目录是开发历史，不能冒充最终运行。

## 真正剩余路径

原程序要得到精确位值闭合，至少需要确定实际 ex2.approx 和 div.full 的位算法/设备绑定，正确复现 max/sum 布局树及存储转换，并维持原全域的 NaN/Inf/±0/subnormal、count=0 和 alias 情形。当前库没有该 SFU 或 div.full 的完整可执行算法；给任意 `exp32` 函数另起名字、选择未经绑定的多项式，或把原程序换成 libdevice/div_rn 后仍称原源都不能解决此缺口。若另行采用一个明确软件实现，必须作为版本修改程序并独立证明，原版本与本反例保留。

本包不改变原 scope、numeric callback 义务或 task 分类。固定反例和实验采集不算完整软件/GPU 正确性。
