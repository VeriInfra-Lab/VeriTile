本目录是独立 v2 修订；v1 保持原样。实际复现了 v1 检查器的两类缺口：删除预选 SM80 的所有记录并同步修改 resolved_plan 后，v1 仍接受剩余 12 条；离线与未运行报告把 environment_binding_complete 伪造为 true 时，v1 也接受。三个伪造报告及实际旧检查结果保留于 `v1-reproduction.json` 和对应 `*-mutated-v1-report.json`，它们都是拒错历史证据，不是有效编译或运行结果。

v2 要求 resolved 架构与事先 requested 架构逐项一致；环境完整标记仅由实际 native execution、精确查询命令、成功退出、匹配 UUID 以及非空 driver/name 重新计算。离线和 NOT_RUN 必须为 false。新参考、24 个离线配置及一次 native 尝试均由修订工具真实重新生成，42 项测试包含上述拒错及原有完整缓冲区/调用校验。CPU test double 仅测试工具流程，不是 GPU 证据。`producer-process-records.json` 记录每个实际子进程、退出码、输入漂移与日志哈希。

`parallel_retention_attention` 的原核算术图审查和完整缓冲区采集工具已完成：**24 种实际离线编译成功，42 项工具测试通过；实际 GPU 尝试为 NOT_RUN，退出码 2，0 次 launch。** 本包没有新增软件位语义定理、没有改变任务分类或任何旧公开声明，工作区、共享对象和冻结父包均未修改。

原接受范围是逐 program 的完整 forward O 和独立 backward DK/DV helper。它包括前向的 off-diagonal 循环、块间衰减和 diagonal 循环，以及后向的完整 descending 循环、post-loop 缩放和 diagonal 循环。当前 `FPCorrect.forward_fp32` / `backward_fp32` 证明完整最终内存等于独立后端递推的写回结果，保留 fp32、每个 cast、`allow_tf32=false` 和 address-dependent undef；`M`、`D` 仍是后端参数。`public-scope.json` 保存两条旧数学主声明和两条当前 FP 主声明的原文本头/SHA，以及全部旧任务文件的 SHA。

DQ、backward shell 的组合、host NK/NV 合并、autograd、跨 program 并发和原生 ABI 不在原接受证明内。本工具也离线编译并可采集原始 backward shell，以观察真实 inlining 后的图；这部分始终标为额外未证观察，不能升级原 scope。不会把 host 的 `sum(0)` 加进已证范围。

实际编译结果是 SM80/SM90 × 两种 head 配置 × 三个入口 × fusion off/on，共 24 项。默认 head 配置 DK=DV=BK=BV=64，ragged/split 配置 DK=160、DV=96、BK=BV=128；前向 BTL=128，后向 BTL=64，BTS=32，warps=4，stages 分别为 3/2。每个配置保存 TTIR、TTGIR、LLIR、PTX、cubin 和编译元数据。它们是原合法配置的样本，不代表旧任意 Nat 维度/步长/程序号均可直接绑定 native i32。

关键发现及数值缺口：

- 原源码 16 处 `tl.dot` 全部是两参数，且显式 `allow_tf32=False`。实际贡献 dot 在 **fusion=false 时仍可合入 live carry**：TTIR 第三参数就是当前或衰减后的累加器。不能把旧 dot-from-zero 后单独 add 直接解释成这个编译图；需要独立完整 DotAcc 递推和实际顺序证明。
- 这 24 个 fp32 图的 PTX 均没有 MMA/WGMMA，实际使用大量标量 FMA。因此不能套用 half tensor-core 的次序模型，也不能仅因关闭 TF32 就断言某个软件 fold 与之相同。
- exp2 使用 `ex2.approx.ftz.f32`。log2 展开为 `__nv_log2f` 的算法，本批 PTX 没有 `lg2.approx`。其总位行为、FTZ、特殊值和舍入对应仍未闭合。
- 所有被审查的 TTIR block load 都有 pointer 与 mask，但没有显式 other。源码未指定 padding，ragged head/time 的值不能改成零；经过 dot、因果 mask 和乘法的全部算术也不能删去，例如 `0 * NaN` 不会因此消失。
- 源模型的 undef 以初态 region/address 为参数；编译中的 undefined/poison/register choice 需要明确关系。完整保存输入分配并不能反推出 inactive load 的寄存器位值。本工具从不以 guard 内存代填 undef，也不把 ragged 输出登记为确定数值参考。

精确逐项工作和实验要求在 `numeric-obligations.json`（7 项）；逐图指令、dot 第三参数、原源码位置和加载信息在 `compiler-observations.json`。这些是绑定实际工件的文本观察，不是正式的编译等价证明。基础 FP32 位算术可用于后续实现，但当前缺少完整原语/编译图对应，所以没有用任意回调包装出新的“闭合”证明。

原始四个 `@triton.jit` 函数均逐 AST 提取核对，源码体完全保留；没有换成供应 decay、log2 或 exp2 值的核。工具不执行原文件 import 时的 CUDA 测试函数。该旧测试函数后几项改变 q 的 head/batch 却复用旧 k/v，存在不一致的实际尺寸/布局；这些是 host fixture 边界，不是本次内核定理修复。采集语料独立明确为尺寸一致的原核调用。

五个完整实验语料分别覆盖：原测试的 B2/H4/T128/DK64/DV64 形状、T256 的完整两前向块及多轮后向、T129 尾部和特殊值、DK160/DV96 的 ragged head 与 NK 分片、非紧密行/批步长。所有输入和输出都有 origin=8 个 fp32 元素。完整保存 Q/K/V/dO 以及原始 O/DQ/DK/DV 分配的 before/after uint32 JSON 和小端二进制，包含两侧 guard、stride holes 和所有 NK/NV partial；**不做 host 合并**。特殊输入包括 ±0、subnormal、±Inf、NaN 位型，不新增 finite/noNaN 假设。

对每个 program 独立列出两段循环的实际次序、全部输出足迹和 load descriptors。后向 helper 直接调用原 `_parallel_retention_bwd_dkv`，逐项记录 i_bh/i_c/i_k/i_v/i_h，并顺序执行语料中全部程序；这个执行方式不伪称已证明原 shell 的并发组合。原 shell 另有独立记录。全语料默认每架构 5 用例 × 3 入口 × 双 fusion=30 个 stage capture；其中 helper 含多次原核 launch，计数逐项保存。使用 `--case` / `--stage` 时明确记录选择范围，不能冒称整个语料已运行。

输出数值比较固定为 **NOT_ESTABLISHED**。工具报告所有 active 输出原始字、完整非写入位置 frame、未指定 load lane 数及每次原核调用；即使 frame 不变也只标 `CHARACTERIZATION_NUMERICS_UNBOUND`，`numeric_acceptance=false`。没有用 Real 数学、任意最终结果或另一个命中 profile 补足数值证明。退出码 0 仅表示编译/采集完成，1 表示编译、调用、绑定或 frame 失败，2 表示没有 GPU 执行。

在本包根目录复核：

```bash
python -B verify.py
python -B src/verify_capture.py offline-v1
python -B src/verify_capture.py gpu-attempt-v1
python -B src/test_runner.py
```

在有 SM80/SM90 GPU 的机器运行完整原核采集，输出路径必须不存在：

```bash
python -B src/run.py run --reference reference-v1 --output /tmp/pra-new-native-capture
python -B src/verify_capture.py /tmp/pra-new-native-capture
```

可显式只采集某阶段或用例，报告会保留这一限制。例如：

```bash
python -B src/run.py run --reference reference-v1 --output /tmp/pra-new-helper-capture --case ragged_time_special --stage backward_helper
```

重复同一 ragged case 可观察是否重现相同输出，但重复吻合不能证明 undef 稳定或普遍原语关系。离线复编命令为：

```bash
python -B src/run.py compile --reference reference-v1 --output /tmp/pra-new-offline-capture --arch 80 --arch 90
```

`final-validation/receipt.json` 绑定正式实际编译/无 GPU producer 报告，再真实运行逐工件核验和 42 项测试；没有把这一步称为重编了 24 个核。运行分支的 CPU test double 覆盖三个入口、完整 helper 调用序列、buffer/physical-origin 检查和错误路径，临时模拟记录未作为 native 收据保留。实际 Triton 包、Python executable、ptxas/libdevice 前后 SHA、GPU可用时的 UUID/驱动查询也被记录。未声称 OS/所有宿主库已 hermetic 绑定。

纯哈希/编译工件/原始数据辅助函数复用冻结 RBE 工具，父 receipt 和辅助文件 SHA 已写入 `inputs/source-bindings.json`；不继承 RBE 的数值模型。所有实际编译工件、报告、日志保留，只有结束后的重复 compiler/bytecode cache 被清理，记录在 `cache-cleanup.json`。v1 父包的 `probe-v1` 为早期探索，本 v2 不复用其执行记录；`offline-v1` / `gpu-attempt-v1` / `final-validation` 是正式结果。
