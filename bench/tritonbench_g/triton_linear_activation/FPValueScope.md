# Linear activation floating-point value contract

`FPCorrect.source_backend_values` covers the complete previously accepted grouped
L2 tile: arbitrary dimensions/strides/group size and K-block count, both
`HAS_BIAS` and `SHOULD_SAVE_ACT_INPUTS` branches, and every activation string.
The accepted specialization is `SPLIT_K=1`, `K_LOAD_MASK_NEEDED=true`, and
`K=BLOCK_K*numKBlocks`. The ascending Lean counter retains the old transcription
of the original descending exact-K loop. Host grid/autotune and the other
split-K/partial-K configurations are outside that existing scope. No additional
finite-value, non-NaN, output-value, or dot-accuracy premise was added.

The reference reads only the independent initial memory. It computes the grouped
program geometry, wrapped row/column gathers, bias cast, every K block, activation
and storage conversions. It does not execute the source or assume its output.
`execution_memory` works without layout assumptions, including zero blocks and
arbitrary aliasing. `Post` gives active output readback, optional saved-preactivation
readback when saved/output regions are distinct and saved addresses injective,
and frame outside both write footprints. Output injectivity follows from the old
fitting-tile and minor-stride-one/major-stride-width conditions; saved injectivity
has the analogous proved bridge. The old IO skin's bounds/allocation assumptions
remain ordinary input conditions, not pending numerical laws.

Three named physical storage formats are represented: fp16, bf16 and fp32, for
input/weight/bias/output, with independent saved format. Half/bfloat payloads occupy
the low 16 bits of the `FP.Value.fp32` carrier, not fp32 numerical encodings.
Register casts explicitly widen loads to fp32 where the source does so; output
and saved stores explicitly narrow. The original Python host requires common
input/weight/bias dtype, and the supplied test/AMP path uses fp16. No fp64 or fp8
storage claim follows. `source_projection` proves the erased operation/control/
address projection for all nine storage/saved combinations. Renaming the formal
bias pointer to `biasR` prevents a frontend pointer/register-name collision; the
numeric register is still `bias`, and the expanded projection is unchanged.

Scalar fp32 add/sub/mul, storage conversions and maximum are concrete software
operations with explicit `FP.Config`. ReLU uses maxnum, including lone NaNs and
mixed signed zero. Host constants are not unconstrained opaque values: the
actual Python binary64 values and fp32 encodings are recorded by the native probe.
`FPLiterals` binds the old Real expression tags to fp32 bits: sqrt(2)=0x3fb504f3,
sqrt(2/pi)=0x3f4c422a, 0.044715=0x3d372713, 0.5=0x3f000000 and 1=0x3f800000.
This does not identify mathematical Real.sqrt with host binary64 evaluation.

The complete activation contract remains a backend-value contract: `Dot` or
`DotAcc`, `tanh32`, `erf32`, and `divide` are explicit full callbacks. Native
libdevice tanh/erf and PTX `div.full.f32` are not claimed equal to real functions
or correctly rounded software division. The dot callback retains all operands,
accumulator where present, dtypes, precision, options and matrix dimensions.
The fallback algebra is unused by the independent reached-value formula.

`compiled_graph_backend_values` separately represents the offline observed
explicit-accumulator dot graph and normalized scalar operand order. Source
zero+bias becomes bias+zero; GELU and fast-GELU operand orders are retained as
observed; ReLU maxnum(zero,x) becomes maxnum(x,zero). No unconditional all-bit
commutativity, fusion or source/compiled graph equivalence is assumed. The native
SM90 dot uses its own MMA/TF32 policy and reduction order, not the software
increasing-K fold by definition.

`scalar_activation_sequential_bits` is an additional, explicitly restricted
ReLU/identity software profile. It closes dot with sequential fp32 mul/add for
the source graph or sequential FMA for the accumulator graph, plus concrete
scalar/cast/maxnum operations. This secondary configuration does not restrict
the two full-activation contracts or upgrade the entire task to a closed bit model.
The positive two-block witness covers nonzero grouped program ID, all storage
formats, both stores and frame; the cancellation witness exercises all five
activation branches. Its zero-only tanh/erf callbacks are deliberately defined
software test callbacks, not native libdevice evidence. Fixed witnesses never
upgrade the parameterized model role.

Offline evidence uses Triton 3.8, CUDA SM90, BLOCK_M/N/K=32, GROUP_M=8 and
`enable_fp_fusion=false`. All five original helper/kernel function bodies are
AST-identical after excluding host/autotune wrappers. Fifteen original profiles
compile; three original fast_gelu profiles fail because `sqrt2pi` is a mutable
global rather than a constexpr. A separate minimal candidate changes only its
existing initialization to `tl.constexpr(...)`, and its three fast_gelu profiles
compile. Candidate success is not original-source success. There was no GPU
execution. Original/candidate sources, diff, host constant bits, complete IR/PTX,
compiler errors and hashes are archived separately.

Outstanding native bindings are profile-specific: scalar rounding/FTZ/NaN and
cast behavior; actual MMA/TF32 dot accumulation; libdevice tanh/erf; `div.full.f32`;
and the original fast_gelu source repair. The compiler's observed source-to-dotAcc
rewrite is not a universal numeric law awaiting random verification. Natural
address arithmetic versus machine-address validity is the existing host/input
boundary, not an extra floating-point experiment obligation.

The old main's maximum AST compatibility patch changes only local implementation
proofs; all 68 old theorem/specification statements are byte-for-byte preserved.
Its independently frozen receipt is included in the final evidence package.
