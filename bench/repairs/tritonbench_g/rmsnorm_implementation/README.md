# rmsnorm_implementation: kernel proofs for the checked FP32 source

`Kernels.lean` is the canonical readable typed DSL. It preserves the original
4096-lane passes, FP32 input/weight/output, first-pass cast, `var += xf*xf`,
reduction, division by width, `sqrt(var + eps)`, uncast second input load,
`x / std` then weight multiplication, and the omitted weight-load fallback.
The candidate changes only `pid_batch * stride_x_batch` to
`stride_x_batch * pid_batch`. `Proofs/Source.lean` connects these displayed
sources to the reusable source proofs by definitional equality and execution
transport. No Python guard, kernel statement, compiler option, or precision
choice is changed by this proof refactor.

`Contract.lean` contains the independent mathematical formula and input
requirements. `Proofs/Contract.lean` proves that its checks match the executable
Python guards, including rejection of non-Tensor arguments before metadata
access. The formula is

```text
output[b,m,k] = input[b,m,k] / sqrt(sum_j input[b,m,j]^2 / K + eps) * weight[k]
```

`Correct.lean` exposes the original-source specification
`Correct.rmsnorm_kernel_correctness`. `Misc/AuxiliaryCorrect.lean` retains the
candidate result `Correct.rmsnorm_index_commuted_correctness` as an ordinary
theorem with its unchanged contract. Both prove successful execution, every
active row output, and the entire memory
frame through the shared `UKernelIO.RegionalScheduledImplements` judgment.
`IO.lean` exposes two finite input channels, one finite output channel, both
program-ID-dependent strided windows, the full FP32 source identity and the
independent finite RMS formula. A singleton identity launch preserves the initial
program IDs/registers/undefined payloads and executes both complete passes.
Declared capacity bounds imply safety of the actual evaluated loads and stores;
`Proofs/IOSafety.lean` reuses the old body walks with active-only injectivity.
`Proofs/IOResults.lean` composes that safety with the existing value/frame proof.
The former operational statements remain unchanged as ordinary theorems in
`Misc/Auxiliary.lean`/`Proofs/RawCorrect.lean`; they are also the common execution
foundation used by the Tensor applications. They quantify over program IDs and arbitrary feature widths. They need
uniqueness only among active output coordinates. In particular, width one with
zero output feature stride is supported: PyTorch permits this contiguous view
and `empty_like` preserves its zero singleton stride. Masked tail pointers need
not be distinct. No positive-stride guard is introduced.

`TensorBridge.tensor_kernel_correctness` binds the same source execution to actual
Torch FP32 Tensor objects. `Proofs/TensorBinding.lean` records their observed
metadata, storage identities, standard strided logical-to-physical index laws,
and actual pointers. The theorem proves the logical Tensor formula, full
allocation-relative physical memory frame, and all three source pointer
expressions. Nonzero storage offsets and input-weight aliases remain supported;
the output is fresh. These API facts contain no execution or numerical result.
The Real theorem interprets exact mathematical arithmetic; it does not identify
FP32 outputs with exact Real results.

`Misc/TensorBridge.lean` also contains the candidate Tensor application
`tensor_index_commuted_correctness` and `python_checks_iff_preconditions`.
These applications are ordinary theorems; the primary entries do not import
the bridge.

`FPEquiv.rmsnorm_index_commuted_equiv` proves that both complete typed FP
executions succeed and agree on every typed memory cell, under arbitrary scalar
backend semantics and arbitrary input/padding payloads. Its numerical rule list
is empty. `Proofs/FPSuccess.lean` establishes successful execution from register
invariants for both loops, so equivalence does not rely on an assumed successful
run. No floating-point reassociation, split/FMA identity, or finite-input
restriction is used.

`TensorBridge.observed_tensor_kernel` separately proves all logical output FP32 words,
actual Tensor pointer/load binding, and the full physical frame for the explicit
observed native graph. That graph uses square-accumulate `fma.rn.f32`, the
supplied complete ordered scalar-add reduction tree, `div.full.f32`, epsilon
addition, `sqrt.approx.ftz.f32`, output `div.full.f32`, and weight multiplication.
`Proofs/KernelValues.lean` exposes these scalar instruction semantics directly.
An optional software interpretation supplies concrete rounded FMA/add/multiply;
literal conversion, `div.full`, and approximate FTZ sqrt remain explicit scalar
ports. They are primitive semantics, not experimentally admitted equality rules
or callbacks returning an entire row.

Compiler correspondence from the typed split source to that observed fused
native graph, the actual reduction tree, and hardware primitive conformance are
explicit external boundaries. Existing offline evidence uses Triton 3.7.1,
CUDA SM90, width 4097, block 4096, 16 warps and default fusion. It does not prove
compiler lowering for every accepted specialization or claim GPU execution.
The source-equivalence theorem and native-graph value theorem are distinct.
`TensorBridge.observed_tensor_kernel` retains the complete Tensor application.
`Auxiliary.observed_kernel` exposes the same observed two-pass native graph over
raw allocation IDs, offsets, strides and geometry. Its layout has no Tensor
objects, shape metadata, device flags or admission checker. Both results reuse
the original row-value/frame proofs. The typed-source/compiler correspondence
boundary remains explicit; this interface change does not add a compiler proof.

The finite-machine and host companions are optional. `Misc/FiniteCorrect.lean`
exports the existing guard-derived signed-i32 intermediate/loop, padded 64-bit
pointer, active four-byte bounds, and output-coverage certificate.
`Misc/HostCorrect.lean` keeps allocation, empty return, rejection, setup failure,
device restoration, and whole-grid host results separate from the kernel
statements. The original loop-overflow defect remains independently documented:
width 2147479553 produces final loop increment 2147483648, which is negative in
signed i32. The checked wrapper requires padded width strictly below `2**31`.

Historical validation (unchanged by this format-only pass):

```sh
python bench/repairs/tritonbench_g/rmsnorm_implementation/check_contract.py
```

The checks pass unchanged kernel AST and guard comparisons, 34 rejection cases,
nine admission cases, and eight exact CPU Triton-interpreter controls. Controls
include partial/multiple chunks, storage-offset views, unusual singleton strides,
zero leading dimensions, the final-loop overflow witness, and the new zero
feature-stride input/weight/output regression. With `x=±1`, weight 2 and epsilon
3, all interpreter outputs equal `x` exactly. CUDA is unavailable. Lean 4.29.0
checks the public proofs with standard-base axiom audits; modules requiring a
larger stack are checked with targeted `lake env lean -s 32768` invocations.

The current format pass runs targeted Lean builds of the public entries, IO
adapters, Auxiliary and TensorBridge, including statement-surface, independent
formula and standard-base axiom gates. The FP numerical assumption report is
`none`. Python/Kernels/Contract and the Tensor theorem statements are unchanged.
This pass does not add independent replay, GPU runs or numerical experiments.

Optional proof entries are in [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/AuxiliaryCorrect.lean](Misc/AuxiliaryCorrect.lean), [Misc/HostCorrect.lean](Misc/HostCorrect.lean), [Misc/Auxiliary.lean](Misc/Auxiliary.lean), [Misc/FiniteCorrect.lean](Misc/FiniteCorrect.lean). Their declaration namespaces, full statements and proof bodies are retained; the primary entries do not import these optional readers.


The source-only cleanup on 2026-10-22 replaces the selected
`rmsnormRepairSource%` whole-source macro with complete ordinary quotations
for `kernel` and the existing `indexCommutedKernel` in `Kernels.lean`. Both
complete functions are checked against the exact previous module by `rfl` in
`.veritile-cache/finish-pending-105-20261021/source-only/rmsnorm_implementation/Preservation.lean`.
The existing general dtype library is reused unchanged. Both full passes,
FP32 loads/casts/storage, split square/add order, padded reduction, epsilon,
square root, division/multiplication, arbitrary omitted weight padding, masks,
strides, domains, bindings, main program pair and optional native/host/Tensor
results are preserved. This changes source readability only; it introduces no
candidate, numerical rule, experiment, host obligation or completion recount.
