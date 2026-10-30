# softmax_reducev: zero-mass prefixes

This separate candidate repairs the admitted length-65 row with logits
`[-inf] * 64 + [0]` and gathered V equal to one. The selected predecessor
returns NaN because its first tile subtracts `-inf - -inf`; the independent
normalized result is one. The predecessor and its saved diagnosis are unchanged.

The only new kernel statement is
`n_e_max = tl.maximum(n_e_max, -340282346638528859811704183484516925440.0)`.
The literal is exactly the least finite FP32 number, word `0xff7fffff`.
Every finite FP32 maximum is at least this floor, so its subtraction operands,
exp, reduction, FMA and final division are unchanged. In an all-`-inf` prefix,
the new maximum is finite and all weights are zero. Carried zero sums cannot
poison a later finite block. A later logit as small as FP32 minimum is still
normalized against itself; there is no artificial underflow from a floor of zero.
The initial state, masks, index arithmetic, block size 64, launch, FP32 storage,
default fusion, one warp and three stages are preserved.

The wrapper is identical to the selected predecessor. It admits all FP32 raw
words, preserves nonzero storage offsets and irrelevant singleton strides,
checks signed-int32 metadata and active right-aligned page indices, mutates the
supplied output, and returns None. No value-finiteness guard was added. Empty
batch/head calls retain the existing no-launch behavior. The prior padded-page
zero and masked-V-zero repairs are retained.

`Correct.kernel_real` uses the shared `ProgramIO.ImplementsOn` judgment for the
complete source grid. `IO.lean` declares typed length/start/page metadata,
active gathered logits and values, every output channel, and the actual source
invocations. The padded logical extent is the finite supremum of arbitrary
per-batch counts; it does not impose `counts <= maximum`. `Targets.lean` states
the independent unshifted weighted sum. Negative infinity contributes zero
mass; a finite logit q contributes exp(q). `PublicIO.RealDomain` retains at least
one finite active logit per row and finite gathered V, including any number of
all-`-inf` leading blocks. Source access safety, success, complete outputs and
the exact exterior raw-cell frame are conclusions of the shared judgment.

The ordinary auxiliary theorem `FPEquiv.kernel_native` in `Misc/NativeCorrect.lean`
covers all raw FP32 words with the independent ordered
native recurrence and exact exterior-memory frame. It preserves physical
replicas, indexed complete reduction trees, the D128 final denominator add order,
D16 shared-writer contenders and local FMA selection. This auxiliary native theorem
uses the same shared IO judgment with the original per-row primitive model and
word observer. Both kernel interfaces expose geometry, arbitrary logical
counts and a complete program order; operational states and execution details
are in `Proofs/`. `KernelNativeResults.local_refinement` retains the same seven
local instruction-conformance laws.

The former complete raw-grid endpoints remain unchanged as ordinary theorems
in `Proofs/KernelRealResults.lean` and `Proofs/KernelNativeResults.lean`.
`IOCoverage.lean` embeds every old finite-batch input into the logical channels,
including unrestricted signed native decoding. `IOCompatibility.lean` proves
that both public targets agree with the old mathematical and ordered native
targets. That earlier IO organization pass did not change the source, Python guard, dtype or numerical laws.

`TensorBridge.tensor_correct` and `TensorBridge.tensor_fp32` retain the complete
checked Tensor applications. Every input read comes from the original supplied
Tensor storage. Results are read through the supplied output Tensor's storage
identity and offset; word and little-endian byte frames preserve the exterior of
the actual output footprint. Python None and the supplied output identity remain
conclusions. `TensorBridge.tensor_constant` derives the constant result through
the actual output Tensor, including one for the saved witness. The shared
foundation is the existing complete `FPValue.correctness` and
`FPValue.execution_memory` source proof; the public IO adds actual trace safety
and composes the same per-program results over the complete schedule.

Exceptional native scope is explicit. With IEEE-compatible local instruction
meanings, an all-`-inf` row with finite V has denominator/numerator zero and final
0/0 is NaN. A +inf logit forms +inf-(+inf) and a NaN logit has a NaN weight;
these cases have NaN normalization and are outside the normalized Real theorem.
Active nonfinite V remains admitted: zero times infinity or NaN is NaN, including
at a zero-weight active key. The kernel does not discard such V operands.
Other nonfinite V results follow the stated ordered native operations; no finite
normalized answer, NaN payload identity, FP-to-Real equality, or error bound is
claimed. The Real model's totalized zero division is not a native claim.

The native model has seven local instruction families: maxNum, subRN, mulRN,
addRN, divFull, fmaRN and exp2Approx. Exp is the actual multiply by log2(e) word
`0x3fb8aa3b` followed by exp2Approx. The selected backend emits max.f32,
ex2.approx.f32, fma.rn.f32 and div.full.f32. Instruction meanings and the selected
compiler/lane layout remain explicit model boundaries, as do ordinary Torch
storage API laws and GPU/runtime conformance. No numerical experiment is a
remaining acceptance requirement. No GPU execution is claimed.

Validation includes the predecessor's 19 admitted and 36 rejected source/host
controls, 15 new actual-AST controls (empty-prefix witness, -1000/minimum-finite
tails, several empty blocks, offsets/singleton strides, unchanged finite words,
and exceptional values), and four fresh actual-JIT-binder SM80 compilations:
D1, D16, D128 and unaligned D16, with 16 TTIR/TTGIR/LLVM/PTX artifacts.
CPU replay verifies strict active pointer bounds and complete backing-storage
frames. It is not a claim about GPU numerical instruction conformance.

`FPEquiv.reducev_grid_equiv` now compares the complete original source with
the second complete body in `Kernels.lean` under the original complete grid IO
(`Candidate.lean` remains a compatibility import). The candidate moves
`off_o` and `out_ptrs` before the final `acc = acc / e_sum`. This prepares the
store address before division and shortens the normalized accumulator's live
range before its store. No performance measurement is claimed. The Python
entry `token_softmax_reducev_early_address` retains the complete original wrapper
and uses `_fwd_kernel_early_address`; original entries and all guards are unchanged.

`AddressSchedule` proves exact full-state source equality and actual trace-safety
equivalence. The entire online loop, its FP32 load/cast boundaries, max clamp,
exp and reduction trees, FMA choices, primitive instruction callbacks, and
store replica observation are unchanged. `AddressIO` lifts those results through
every complete original workset permutation. The main `≡[[]]` judgment concludes
both source safety traces, both successful executions, all typed output cells
and both exterior frames. It retains arbitrary original FP32 words and arbitrary
per-batch counts; capacity remains derived from counts and is never narrowed to
`maximum`. There are no numerical rewrite rules or new deferred experiments.

The independent normalized WithBot Real target and existing optional native/host
results retain their original scopes. Main entries and their helpers have no
`Misc` dependency. The source is complete ordinary Triton, with no generated AST
or custom runner. The candidate's only change is the two exact instruction moves.

## 原始函数体与 Lean 入口

`Kernels.lean` 的 `FPValue.sourceBody` 和 `FPValue.earlyAddressBody` 分别直接粘贴
`_fwd_kernel`、`_fwd_kernel_early_address` 的完整函数体，包含在线循环、最小有限值
钳位、全部 mask/padding、最终写回及 `return`。`Candidate.lean` 是兼容导入入口。
候选只把 `off_o` 和 `out_ptrs` 提到最终除法之前；两个原始 Python 函数均未修改。

类型和调用参数在函数体外提供：FP32 logits/value/output、INT32 page/start/length，
以及原有非负 start/length 的数学内存视图。类型声明不会成为 Python 语句。
源码没有显式 FP32 load/store cast，因此 Lean 引用也不添加这些 cast。存储通道到
算术通道的转换由前端处理；直接复制保留原存储通道，变量执行算术后不会再被误当作
未经转换的存储值。`e_sum = 0.0` 按 Python 标量赋值生成 FP32 值；load 的 padding
转换保留独立的存储边界。执行证明重新连接这些实际节点，没有假设任意 cast 是恒等。

`jit-correspondence.json` 检查两份完整函数体和调用绑定。`Proofs/FPSource`、
`FPPrefix`、`FPLoop`、`FPContract` 对实际程序分解并证明整个在线循环；
`IOSourceSafety` 覆盖实际访存，`AddressSchedule` 连接完整候选执行和安全性。
公开 Real 公式、FP 程序对、输入域、完整 grid 与外部内存保持条件不变。
前述历史 CPU/backend 证据针对原有 Python 与数值配置；本次未重新运行 GPU 实验。

`Proofs/PrecisionProfile.lean` 将迁移前后的有序递推在实际 NativeGraph 和 RealModel
解释下逐值连接，尺寸、计数和输入均保持任意；独立数学公式与定义域保持不变。

复核入口：

```sh
python3 -m bench.jit_correspondence bench/repairs/tritonbench_g/softmax_reducev_neginf_checked/jit-correspondence.json
lake build bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Correct bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.FPEquiv bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.PrecisionProfile
```
