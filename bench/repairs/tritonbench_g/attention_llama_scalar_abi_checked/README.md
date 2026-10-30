# attention_llama: scalar ABI checked candidate

Current FP pair (2026-10-21): `FPEquiv.attention_kernel_equiv` compares the
actual complete runtime-scalar source with the existing carry-first normalizer
candidate. The source change is `l_curr = tl.sum(p, 1) + l_prev` to
`l_curr = l_prev + tl.sum(p, 1)`, in both FP16 and signed-byte E5M2 paths.
The carry-first candidate already exists in the preserved `attention_llama`
repair; this task binds it to the selected raw scalar ABI and repaired masks.
Its optimization purpose is to expose the rescaled carry product as the
leading normalizer operand for multiply-add selection. No performance gain is
claimed. The fixed source-selection model lowers both spellings to exactly
`fma(old_l, alpha, reduced_p)` with identical operand order, so the theorem
uses `≡[[]]` without add commutativity, a finite-word restriction, or an
experimental numerical rule.

`Proofs/WholePair.lean` fixes the original typed memory interface. Its
implementation data contains only actual invocations, pure native observers
and geometric write ownership. Its equivalence relation requires both
`OrdinaryRun` witnesses from the same initial memory, safety of all actual CTA
runs, equality of every active typed output cell, and both exact exterior
frames. The complete two-dimensional physical grid keeps the existing parallel
disjoint merge semantics. `Proofs/Carry*.lean` proves the candidate source
execution, recurrence, access safety, native lowering and grid connection.
Floating payloads remain arbitrary words, including nonfinite values and
signed zeros; only reached E5M2 input roles require signed-byte representation.
The raw scalar specialization, lane-selected reductions, separate carry
multiplication, normalizer FMA, seeded MMA, half casts, all masks, strides and
outputs are retained. `Misc/NativeCorrect.lean` preserves the former one-source
native theorem as an optional result.

This separately named candidate adds one launch-path check to the existing
masked attention repair: a raw Python integer scale must satisfy
`-(2**63) <= sm_scale < 2**64`. The check follows the empty-output return. The
kernel body, input checks, address checks, raw argument passed to Triton, and
original finite Python-float domain are unchanged. In particular, finite
Python floats larger than the FP32 range remain admitted and convert through
the ordinary FP32 parameter ABI. No floating-range or result-finiteness check
has been introduced.

The actual scalar specializations are represented separately:

| Raw argument | Triton specialization | Numerical interpretation |
| --- | --- | --- |
| integer `1` | constexpr | the score-scale multiply is eliminated |
| remaining signed 32-bit integer | i32 | direct signed-word to FP32 conversion |
| remaining signed 64-bit integer | i64 | direct signed-word to FP32 conversion |
| integer from `2**63` through `2**64-1` | u64 | direct unsigned-word to FP32 conversion |
| ordinary finite Python float | FP32 parameter | binary64 payload converted to FP32 |

`Contract.lean` specifies the sole new check and derives the original input
contract. `Proofs/ScalarABI.lean` proves that launch admission is equivalent
to successful scalar specialization. Its exact binary64 decoder also binds the mathematical scale
to the same raw argument; native conversions do not pass through that decoder.
The empty path never requires scalar specialization.

`Kernels.lean` contains four ordinary readable DSL quotes: both complete dtype
paths for the original and the carry-first candidate. The scalar uses a
nonexecuting `tl.param sm_scale : tl.float32` compute-carrier declaration;
the actual raw ABI specialization is supplied by the invocation. No dummy
assignment or statement-list deletion constructs the source.
The bound scalar register distinguishes the actual ABI operand from source
literals. The shared source includes the preamble, causal/noncausal dynamic
range, masked K/V loads, both dots, explicit half conversions, pointer
advances, and final masked store. Source statements are not replaced by a
handwritten AST. `Proofs/Execution.lean` establishes successful execution,
active stored cells, and every complementary typed-cell frame.

`Target.lean` defines the independent native result from logical input words
and scalar specialization before any execution. It records the saved Triton
3.7.1 CUDA SM80 profile: 64×64 tiles, head sizes 16/32/64/128, four warps through
D64 and eight for D128, two stages, ordered reduction trees, separate exp
multiply/ex2 operations, a fused normalizer update, separately rounded carry
rescaling, seeded K16 MMA composition, and the E5M2 fragment-coordinate
permutation. Input and output dtype conversions remain explicit. No
finiteness, magnitude, underflow, overflow, NaN, or signed-zero equivalence
assumption is used to identify native arithmetic with Real arithmetic.

`Misc/NativeCorrect.lean` states the shared `OrdinaryProgramIO.ImplementsOn` judgment
over finite logical arrays. `NativeIO.lean` names the actual source, tagged
scalar register argument, full physical grid, native lane observer and geometric
write ownership. `IOLayout.lean` exposes the typed windows, strides and masks;
`Mathematical.lean` supplies the independent native formula. Source success,
actual load/store safety, all output cells and the exact frame are conclusions.
The full Tensor application remains `TensorBridge.attention_native_tensor`. `Backing.Objects`
records the caller's Tensor object/storage identities, allocation bases and
sizes, data pointers, dtypes, devices, shapes, and strides. Every input
observation reads one shared caller heap through those pointers. There is one
source result per physical grid program. The parallel grid merges its complete
disjoint store footprints directly from final typed source cells into that same
heap; the allocated output object reads the resulting heap. The theorem proves
full program coverage, output address ownership, full source frames, preservation
of all input allocation cells, and every exterior cell. It has no assumed
successful execution, attention-output equation, whole-graph equality, or
local numerical-agreement premise. The scalar operations, conversions,
exp2 approximation, fixed K16 MMA instruction, and recorded compiler selection
are primitive/compiler model boundaries; they are not deferred numerical
experiments.

The internal source view supplies valid signed-byte reads, while its caller
premise concerns only reached K/V cells on the E5M2 path. Unrelated caller
storage remains arbitrary and is preserved. `TensorObservation.lean` proves
the descriptor-based FP16 and byte observations from the shared initial heap.
The empty-output result preserves that entire heap and bypasses raw scalar
specialization.

`Correct.lean` exposes the FP16/E5M2 mathematical result through a shared
parameterized Algorithm IO judgment. `RealLayout.lean` and `RealIO.lean` name
the finite windows, zero-undef convention, runtime `sm_scale` binding and
complete ordinary grid; `RealMathematical.lean` defines attention independently.
`Proofs/RealSourceTransport.lean` proves execution and access-safety transport
from the actual runtime-bound source to the existing inline-scale proof under
the scalar-register invariant. The public source remains the actual selected
source. The complete shared-caller-backing application is retained as
`TensorBridge.attention_tensor_backing`.
`Misc/TensorBridge.lean` provides the additional checked access/address applications
`TensorBridge.attention_fp16` and `TensorBridge.attention_e5m2`. It reuses complete Real source execution proofs of the identical
unchanged kernel body. The backing theorem merges final source memory cells,
with no independent encode-and-scatter output memory. Its idealized arithmetic target is the existing
masked softmax attention over `k < N`, additionally
`k + start_position <= query` in causal mode. It does not assert equality
between native rounded attention and that Real target. The original causal
and finite-input checks remain in force.

The saved evidence under
`.veritile-cache/task-deliveries/attention_llama_scalar_abi_checked/` retains
the 20 wrapper guard cases, 10 scalar binder transitions, extreme finite-float
specializations (including `1e100`), and compiled SM80 artifacts for the
constexpr-one branch. These checks were not rerun during proof completion.
The saved AST comparison checks both complete readable DSL paths against the
specialized Python body, with the implicit FP16 store conversion made explicit.
Lean modules have task-scoped workspace build receipts. The original
`attention_llama` diagnosis and source files remain preserved. Global
`grid[0] * grid[1]` admission and host/compiler/resource verification remain
outside this candidate's kernel/wrapper scope.

`Correct.lean` exposes the Real IO specification; `Misc/NativeCorrect.lean` retains the ordinary native IO theorem. Operational
execution, state, cell and loop proofs live in `Proofs/`; the former raw grid
endpoints remain ordinary theorems in `RealGridResult.lean` and
`NativeGridResult.lean`. `Misc/TensorBridge.lean` retains all complete shared-backing
Real/native results, including `TensorBridge.attention_empty`. Neither primary
imports the bridge, and no supporting proof imports a primary reader.
`Misc/PublicAudit.lean` audits both interfaces as an optional reader. The pair completion adds no numerical assumptions or experiments. The
Python wrapper and input contract are unchanged; the Lean source now uses
explicit parameter declarations and includes the full existing candidate.

Optional proof layout (2026-10-21): `Misc/NativeCorrect.lean`, `Misc/PublicAudit.lean`, `Misc/TensorBridge.lean` are optional entries in `Misc/`. The required `Contract.lean`, `Correct.lean`, `FPEquiv.lean` and their proof helpers do not import these entries. Declaration names, statements, proof terms and acceptance status are retained; this move adds no mathematical completion or numerical experiment.

Additional optional reader paths: `Misc/NativeGridResult.lean`. Their declaration bodies are retained. Historical readers remain separate from current primary/auxiliary checks.

Campaign validation is recorded under
`.veritile-cache/finish-pending-105-20261021/attention_llama/`. The focused AST
check compares all four complete quoted paths with the selected Python body
and the sole stated carry-first change. Targeted Lean builds recheck the
original Real/native and optional Tensor entries together with the candidate
whole-grid pair. Historical saved compiler and wrapper checks retain their
original scope; no GPU experiment is needed for this structural pair.
