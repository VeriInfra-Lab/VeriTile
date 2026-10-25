# Checked FP32 RoPE embedding

The current public proof object is the complete `_rope_embedding` kernel.
The selected Python revision is unchanged in this kernel revision. It keeps
the original forward `transpose(1,2).clone()`, squeeze and reshape, and the
original backward transpose/reshape. There are no added copies, layout
conversions, dtype alternatives or arithmetic optimizations.

For `H = floor(D/2)`, a lane reads `a = Q[row,head,j]`,
`b = Q[row,head,j+H]`, `c = cos[row % S,j]`, and `s = sin[row % S,j]`.
The independent forward target is `(a*c-b*s, b*c+a*s)`; the backward target
is `(a*c+b*s, b*c-a*s)`. The backward map is the adjoint for arbitrary supplied
coefficients; no unit-circle assumption is made. If `D` is odd, the last
channel is unchanged.

The supported payload remains contiguous strided FP32 tensors on one CUDA
device with resolved view flags. The original input shape is `[B,heads,S,D]`;
the kernel receives its packed `[B*S,heads*D]` view. Coefficient views must be
`[L,H]`, or `[L]` when `H=1`, with `L>=S` and packed rows. Writable Q storage
must be separate from coefficient storage; the read-only tables may alias.
Existing assertions check storage containment/alignment, backward cached
geometry, signed-i32 metadata and padded offsets, and 64-bit formed pointers.
Empty payloads and `D<2` submit no kernel. `Request.Launched` expresses that
boundary without adding a host-execution proof requirement.

`Kernels.lean` contains complete readable typed FP32 transcriptions for the original and
existing index-commuted candidate, retaining both
branches, the four-head dynamic loop, both load-before-store halves, casts,
masks and stores. `selected_source_binding` proves exact equality with the
retained complete typed source. `Contract.lean` owns the independent polynomial,
launch domain and kernel metadata checks. `Correct.lean` presents the actual
source through the shared `ProgramIO` relation and the independent rotation
formula. It proves actual source access safety, successful execution, both
outputs for every head in a group and the full memory frame. The common typed
Q/C/S channels, group masks and stride windows are displayed in
`fast_rope_embedding/IO.lean`; the complete source safety proof is shared in
that task's `Proofs/AccessSafety.lean`. `Proofs/RealResult.lean` retains the old
raw execution endpoint and complete coordinate coverage. `Misc/TensorBridge.lean` contains the checked
Tensor applications, whose checked theorem includes bounds for
formed pointers and masked padding lanes. Helpers also prove signed-i32
interpretation, odd-tail preservation and separation of different programs.

`Misc/NativeCorrect.lean` preserves the ordinary source-to-native theorem for the
same actual source and memory IO, through the shared `ProgramIO` relation. It proves
source access safety, successful execution, exact typed output leaves and the
full exterior frame; every FP32 word remains admitted. The prior whole-word
memory equality is preserved in `Proofs/KernelFP.lean`. It instantiates the
retained source-specific lowering model explicitly: products are coefficient
first; the first output is `fma(c,a,neg(mul(signed_s,b)))`, and the second is
`fma(signed_s,a,mul(c,b))`. Backward `signed_s` is `sub(+0,s)` in the observed
compiler profile. The product tags in the interpreter track contraction in
registers and are removed at memory stores. `softwarePrimitives cfg` fixes
primitive rounding, FTZ, NaN, cast and FMA semantics. These primitives and the
source-to-native compiler lowering are model boundaries. There are **no
numerical transformation assumptions or deferred experiments** for this
unchanged arithmetic graph. No universal FP32-to-Real equality is claimed.

The mathematical and native Tensor applications in `Misc/TensorBridge.lean` start from actual `TensorMetadata.dataPtr`
observations. `Proofs/KernelBinding.lean` derives logical Q/coefficient reads
using shared `Torch.FP32.TensorLaws`, including nonzero storage offsets,
contiguity and the absence of the negative-view flag. `Proofs/RealBinding.lean`
constructs the Real source state from those same physical input reads under
an explicit `Word → Real` input interpretation, then substitutes the derived
logical values into the independent polynomial. The checked Real endpoint
therefore uses the same pointers for source inputs and finite address bounds;
it does not accept an unrelated Real memory state. The input interpretation
is not an assertion that FP32 arithmetic is exact over Real. No output values
or execution results are premises of either binding. Prior optional
host/returned-Tensor proofs remain in `Misc/HostCorrect.lean` and their
original helper modules.

The original unguarded source is not certified by this revision. A concrete
counterexample uses `[B,heads,S,D]=[1,2,3,4]`, cosine `[3,2]`, and sine `[1,2]`.
Its cosine-only length assertion passes, squeeze leaves the sine with two
words, and row 2 reads beyond that storage. The CPU source replay reproduces
the invalid read; the selected checked revision rejects before launch.

Validation uses targeted Lean builds and standard-base axiom audits; the
existing CPU suite checks the unchanged JIT/settings AST, eleven forward and
backward storage paths, fifteen rejection/cache paths, and the counterexample.
Four offline Triton 3.7.1 compilations (SM80/SM90, both directions) check the
complete ordered native output graph and six masked FP32 memory accesses.
Those finite compiler observations are not a compiler-correctness theorem.
CUDA is mocked for CPU replay. No GPU execution or independent native numerical
replay was performed.

The optional applications are `TensorBridge.rope_embedding_tensor_correct`, `TensorBridge.rope_embedding_checked_kernel_correct`, and `TensorBridge.rope_embedding_tensor_fp32`. Neither primary entry imports the bridge.

## Program-pair review, 2026-10-21

`FPEquiv.lean` now compares the actual complete FP32 source with the existing
`fast_rope_embedding` index-commuted candidate. The exact source binding is
retained in `Proofs/SourceBinding.lean`; `Proofs/Pair.lean` reuses the existing
full-loop proof. Only the natural-number head-group product changes. Both
rotation directions, casts, arithmetic order, input domain and stores remain
in the comparison. The experimental rule set is empty.

The historical `rope_embedding_kernel_fp32_equiv` source-to-native result is
preserved as an ordinary theorem in `Misc/NativeCorrect.lean`, imported explicitly
when needed. It is auxiliary to the program equivalence.


## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.

Source readability maintenance (2026-10-22): the retained `indexCommuted`
candidate is now a complete ordinary Triton quotation beside `rope`, preserving
the sole integer head-group operand change. Both full functions are
definitionally equal to the previous definitions for all parameters and both
directions. The source-only cache records preservation and fresh joint
main/retained optional imports; source domains, precision and judgments remain
unchanged.

### 函数体来源与重新验证（2026-10-25）

`Kernels.lean` 逐句粘贴所选 `rope_embedding_repaired.py` 的函数体，保留
两次 `.to(sin1.dtype)`、前后向分支、动态 head 循环及两次写回。
运行 strides/seqlen 与调用 dtype 在函数体外绑定；`ROPE_GROUP_SIZE`
是原函数的 constexpr 参数（默认 4），去掉旧 Lean 多出的赋值。
既有公开证明仍取组大小 4，并直接对新函数体重证全部执行、输出、
安全、frame、Real/FP、native/host/Tensor 范围。两份实际完整程序的
FP 等价通过整数索引交换证明，不再借用含多余赋值的旧程序相等。
`index_commuted.py` 明确是为已有 Lean 索引候选新写的 Python 对照，
不是恢复出的历史原始代码；原始 Python 未修改。

普通 RoPE 的公开 Real 与 native IO 同时改为实际普通 RoPE kernel；
共享的只有唯一内存映射和逻辑坐标双射。`Proofs/Interface.lean` 对这个
真实调度重新证明完整输入、执行、安全、输出及 frame，避免入口仍在
验证另一道 Fast RoPE。IO 实例集中于 `IO.lean`。
