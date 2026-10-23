# layer_norm_ops: kernel-only FP32 source correctness

The candidate retains both source kernels, all optional source branches, the
six autotune warp choices, default fusion, and the host gradient sums/casts.
The original benchmark Python is unchanged. Kernel repairs are limited to
independent Mean/Rstd strides in backward and zero-filled backward W padding.
The new host checks make the selected input domain explicit.

## Mathematical target and precision

Let `z = x + residual` when residual is supplied, otherwise `z = x`. For each
row of width `N`, set `mu = sum(z)/N` in LN mode and `mu = 0` in RMS mode:

```
v = sum((z - mu)**2) / N
rstd = 1 / sqrt(v + eps)
h = (z - mu) * rstd
y = h * weight + bias          # omit bias when absent
g = dy * weight
c1 = sum(h * g) / N
c2 = sum(g) / N                # c2 = 0 in RMS mode
dx = (g - h*c1 - c2) * rstd + dresidual
dw[j] = sum_rows(dy[row,j] * h[row,j])
db[j] = sum_rows(dy[row,j])     # only when bias is present
```

The extra `dresidual` term is omitted when absent. Backward consumes the `z`,
Mean and Rstd actually returned by forward in the composed use. Recompute
stores the same source affine output. With the selected FP32 dtypes, a requested
residual gradient aliases `dx`; no separate dresidual allocation is needed.
Each backward program stores the DW/DB sums over its assigned increasing-row
interval. The retained host wrapper then reduces those program partials.

The selected input/output/accumulator precision is FP32, including statistics,
partial dW/dB buffers, host sums and final casts. Every original cast and every
sequential row accumulation remains. The primary native graph retains source row order, selected operand order,
FMA contraction choices and complete ordered kernel reduction plans.
It does not identify different reduction orders or replace FMA by two rounded
operations. The math definitions in `Proofs/Logical.lean` are independent of
source execution.

The source row-gradient formula has the conventional derivative interpretation
for finite real data and positive epsilon. The retained exact AD experiments
were bounded evidence; this package does not claim a general Lean calculus
theorem or an FP-to-Real error bound.

## Runtime contract

- Inputs are strided, canonical contiguous, offset-zero FP32 tensors on CUDA
  device 0, with no unresolved negative view flag. Nonempty data pointers have
  four-byte alignment. Read-only inputs may alias.
- Matrix shapes are `(M,N)`, strides `(N,1)`, with `M >= 0`, `1 <= N <= 16384`.
  Weight and optional bias are `(N,)`, stride `(1,)`. Residual/dy/dresidual
  matrices have the same shape as x.
- Current CUDA device is 0. This preserves the source's `device="cuda"`
  statistics allocations before its original per-launch device context.
- `eps` is exactly a builtin finite positive Python float. The default is
  preserved. Zero, negative zero, nonfinite scalars and other Python scalar
  types are rejected. Positive binary64 epsilon may still underflow when
  converted to FP32; no nonzero native denominator is assumed.
- Boolean flags have exact bool type. `out_dtype` and backward `x_dtype` are
  None or FP32. `residual_dtype` is None or FP32 when residual is absent; when
  residual is present, its dtype overrides that argument exactly as in source.
- Plain backward requires Mean; RMS backward requires Mean=None. Rstd is
  always required. Produced Mean/Rstd are `(M,)`, contiguous and offset zero.
  Standalone arbitrary strided statistic views and plain missing-Mean behavior
  are outside this selected host contract.
- `BLOCK = min(16384, next_power_of_2(N))`. Padded source bounds are
  `M*N+BLOCK < 2**31`, `sm_count*rows_per_program*N+BLOCK < 2**31`, and
  `sm_count*N+BLOCK < 2**31`. `sm_count` is the actual positive CUDA property.
  The retained `math.ceil(M/sm_count)` result is checked against integer ceiling
  division before allocation. This connects the actual source grouping to the
  proven row partition without assuming a host floating-point result.

`forwardCheck_iff` and `backwardCheck_iff` prove the exact admission conditions.
The records include ordinary API observations of current device, SM count and
the retained row-width computation. Source shape/stride/branch parameters are
constructed from those checked records. `Checked.supported` derives the entire
latest reviewed `Scope.Supported`; it is not an additional unchecked metadata
premise.

The two functions remain separate APIs. `CompositionBinding` describes the
ordinary call sequence that passes forward's returned z/Mean/Rstd into backward
with matching shape, mode, weight/bias and residual flag. Its fields contain
descriptor/region relationships, never desired statistics or output values.
The proof executes the forward producer before backward. The standalone
backward source theorem remains available for caller-supplied statistics.

## Source and host repairs

Forward kernel arithmetic is unchanged. Backward adds `stride_mean` and
`stride_rstd` independently to the signature and respective loads; the wrapper
passes the actual strides, including zero for absent RMS Mean. Backward W uses
`other=0.0`: a non-power-of-two width otherwise permits undefined/NaN padded W
values to contaminate `w*dy` and the row reductions despite zero-padded dy.
The included CPU source control reproduces that permitted undefined-tail
failure; it is not a measured GPU failure.

The source's allocations, epsilon arithmetic, cap/branch logic, device context,
recompute behavior, `_dw.sum(0).to(weight.dtype)`, optional dB sum/cast, and
returned dresidual alias all remain. The SM property query is moved into the
pre-allocation check and reused, rather than executed twice. All six original
warp configurations remain attached to each kernel.

## Primary proof boundary

The primary public entries are kernel-only. `Correct.lean` exposes
`layer_norm_ops_forward_kernel_real` and `layer_norm_ops_backward_kernel_real`.
`FPEquiv.lean` proves two complete implementation pairs: forward column-mask reuse
and backward row-start reuse. Both use the empty experimental rule set. The prior
selected-object `*_kernel_equiv` results are optional ordinary theorems in
`Misc/NativeCorrect.lean`. The general primitive-graph results remain ordinary
theorems in `Proofs/KernelNative.lean` and `Proofs/IONative.lean`.
The optional `Misc/TensorBridge.lean` contains the corresponding `*_tensor_real`
and `*_tensor_equiv` applications as ordinary theorems, plus
`TensorBridge.forward_checks_iff` and `TensorBridge.backward_checks_iff`.
Neither primary entry imports the bridge.
Historical composed-host proofs are preserved under `Proofs/HostCorrect.lean`
and `Proofs/HostFPEquiv.lean`; the optional native theorem makes no claim about
the host `_dw.sum(0)` and `_db.sum(0)` implementations.

`Correct.lean` and `Misc/NativeCorrect.lean` display the shared `ProgramIO` fields, actual selected source,
FP32 channels, strides, masks, aliases and empty scratch list. `IO.lean` shares
the channel definitions. `Correct.lean` states `io ⊨ formula` for independent
population-normalization and row-gradient formulas. `Misc/NativeCorrect.lean` uses the
shared `ProgramIO.Observes` judgment for the existing native model. Neither
statement contains an initial/final State, raw runner or per-address postcondition.
The scope remains one complete forward program and one complete bounded
backward row-block loop; this update does not introduce a new full-grid theorem.

`KernelExecution` constructs successful execution of the actual typed source
kernel, using the retained four forward cases and eight backward loop-body
cases. The same constructed final state has every active output and a complete
cell frame. Forward covers Y, optional residual output, Mean and Rstd.
Backward covers DX, optional recomputed Y, the source's separate residual-store
branch, and every partial DW/DB lane. The checked FP32 wrapper selects no
separate residual store; its returned residual gradient aliases DX. Idle
backward programs still store their zero partials. Forward with M=0 has no
launched programs.

`KernelReference` defines source-ordered scalars and an increasing-row scalar
fold using only the initial input snapshot. `KernelReal` relates these to
independent population-normalization and row-gradient formulas and finite sums.
No execution-success, saved-statistic-correctness, desired-output or
whole-reduction-equivalence premise is used.

`KernelNative` builds symbolic instruction trees from the typed source. FP32
storage and inferred FP32 arithmetic are lowered before the structural
instruction interpreter chooses operand order and adjacent FMA contraction.
The shared reduction plan proves every lane occurs exactly once; neutral
leaves are explicit. Compiler choices inspect shapes without numerical input
payloads. All admitted FP32 words remain in scope, including infinities, NaNs,
signed zero and subnormals. Primitive rounding, FTZ, sqrt and NaN behavior are
explicit target-model definitions, not deferred transformation experiments.

`KernelCompiler` binds the graph to the same selected object bytes, exact
candidate source hash, pointer dtypes, integer slots, independent statistic
strides, branch flags, original warp configuration and compiler defaults.
Epsilon starts as the actual builtin binary64 host scalar. Its conversion law
identifies the scalar leaf with the selected ABI word, and the forward proof
uses that word in the independent output and saved-inverse references. A
positive host epsilon may underflow to zero. No positive native denominator is
assumed.

The count denominator uses the original runtime signed-i32 `N` argument,
converted to FP32. The binding identifies forward integer slot 4 and backward
slot 9 and their signed-i32-to-FP32 recipes. A separate scalar-only conversion
law connects the source model's real count literal to this runtime conversion;
integer range checks alone do not supply that equality. Source observations,
independent native targets and Tensor readback all consume the converted word.
The resulting `CountFacts` prove the bridge for every observed cell and retain
the division node even when `N=1`. The bound decoder schedule disables literal
unit-division elimination, matching the retained runtime `sitofp`/`divf` in
the offline width-one profiles. Coinciding epsilon, count and unit literals are
covered by equality of the complete instruction records.

`KernelInputs` derives views from the exact guards and ordinary Tensor API
observations. `KernelCoverage` proves every active source read is inside those
original tensors. Read/read storage aliasing is unrestricted. The Tensor
entries run the typed source from actual pointer reads, relate those reads to
original logical elements, and attach all output stores to actual Tensor
addresses using local four-byte memory laws. Fresh output allocations preserve
live input reads, and every byte outside the complete output footprint is
unchanged. These are ordinary Tensor/allocator/compiler model interfaces; no
PyTorch or driver implementation is claimed verified.

The original reviewed candidate provenance remains
`2026-10-18/layer-norm-ops-whole002-repair-integration/repair-review.json`,
SHA256 `e17a1c266f8613524d192893c9996c4222eda4cddfda40eaf1c07bd0ab4f509c`.
The current canonical source is unchanged in this kernel-only delivery:
`a04aa0d8020ac89ab5cd5b5798736532685a8734535e2085f7f81e942d56f795`.
No historical acceptance count is copied.

## Validation

The retained kernel dependency builds and the current public IO build use
exact-source receipts from the workspace Lean driver. The current helpers,
both main files and complete TensorBridge have targeted receipts under
`.veritile-cache/public-io-format-20261020/batch06-layernorm-*.log`.
Public entries contain standard-base axiom and statement-surface audits,
and independent Real/native references contain
noncircularity audits. The allowed axiom base is Lean's standard
`propext`, `Quot.sound`, and `Classical.choice`.

`python bench/repairs/tritonbench_g/layer_norm_ops/check_contract.py` passes
exact kernel/host source checks, 30 rejected calls, 16 admitted composed CPU
source executions, and the original-tail failure control. Cases include
LN/RMS, residual and bias options, recomputation, M=0, N=1, padded widths,
idle programs, saved-statistic flow, host sums/casts and returned aliases.
CUDA is mocked; these checks are not GPU numerical runs.

`.veritile-cache/layer-norm-ops-kernel-profiles/reproduce.py` compiles the
unchanged selected JIT bodies offline with Triton 3.7.1 for SM80. Its 16
successful profiles cover all six warp choices for both padded-width kernels,
plus width-one and maximum-width controls. TTIR, TTGIR, LLVM IR, PTX and cubin
artifacts are retained with hashes. Only decorator wrappers are replaced by
`jit`, with the original flags supplied explicitly. These bounded profiles
check compiler acceptance and inspect instruction structure; the Lean compiler
binding remains the explicit model boundary for a selected object.

The strict kernel interface keeps all checked-call consequences in the bridge:
`TensorBridge.admitted_empty_backward_partials`, `admitted_forward_indices`,
and `admitted_backward_indices` accompany the existing Tensor applications.
The primary Real/native theorem types expose region/stride/shape parameters,
the logical IO and independent targets. Their admissibility records retain
the existing region separation, width and source control conditions. Native
`Binding` still describes selected compiler bytes, scalar ABI and source
geometry; it contains no Tensor API object or guard.

`Proofs/SourceSafety.lean` follows every actual optional branch and every
backward loop successor. `IOSafety.lean` derives those access bounds from the
public windows, including the backward weight load before an empty loop and
the final zero partials. `FiniteReference`, `ForwardIO` and `BackwardValues`
connect logical arrays to the existing source-ordered scalar results;
`BackwardIO` retains every optional output and the exact exterior frame.
`Mathematical` and `MathValues` reuse the existing Real formula proofs;
`IOResults`, `IOReal` and `IONative` assemble the shared judgments. Native
targets are independent of the source, including transitively through IO.
The unchanged TensorBridge continues to use the same complete execution
foundation. No Python, kernel quotation, input contract, numerical law,
historical acceptance, independent replay or GPU status changes in this pass.

## Selected pair completion (2026-10-21)

`Kernels.lean` retains the original full ordinary `triton` forward and backward
bodies and adds two related optimization candidates. The forward candidate
computes `mask = cols < N` immediately after `cols`, then reuses it for input and
residual loads, the residual store, both variance branches and the final affine
stage. The backward candidate changes
`min((row_block_id + 1) * rows_per_program, M)` to
`min(row_start + rows_per_program, M)`, reusing the multiplication already
performed for `row_start`. These remove repeated integer comparisons or a
multiply; no performance measurement is claimed.

The public `forward_mask_reused_equiv` and
`backward_row_start_reused_equiv` compare actual complete programs through shared
`ProgramIO.Implementation`. Source execution and access-safety proofs preserve
arbitrary floating payloads, every flag, all outputs and both exterior frames.
The backward proof uses strict natural-number distributivity; the forward proof
uses exact mask reuse. Neither pair rewrites a floating operation or requires an
experimental numerical assumption. The original admissibility conditions and
single-invocation / bounded-row-loop scope are unchanged.

The independent Real `Correct` targets and optional single-source
`Misc/NativeCorrect.lean` results remain at their previous scopes. Historical
native/compiler numerical obligations are separate from the two main pairs.

Optional native observations, Tensor/host applications and reader/audit entries
are in `Misc/`. Import these modules explicitly when needed; the primary
`Correct.lean` and `FPEquiv.lean` entries do not re-export optional results.
