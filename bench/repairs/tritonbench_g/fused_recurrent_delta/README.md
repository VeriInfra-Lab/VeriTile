# Fused recurrent delta: kernel proof

`Correct.lean` exposes shared `ProgramIO.Implements` / `ImplementsOn`
judgments for the complete FP32 runtime forward kernel and backward VJP. `Misc/NativeCorrect.lean`
retains ordinary shared `ProgramIO.Observes` theorems for the two complete native
kernels, under their original full names. `FPEquiv.lean` proves the actual original/candidate runtime program pair;
optional native results require an explicit Misc import. The full Tensor applications
and `kernel_physical_logical_input` remain ordinary theorems in
[TensorBridge.lean](Misc/TensorBridge.lean), imported explicitly by the task audit.

`Whole.lean` exports kernel-entry correctness for the complete readable forward
and backward source in `Kernels.lean`. Each main theorem executes one actual
source program and proves independent mathematical or native-word results
with the complete memory frame. The bridge connects those results to shared
`VeriTile.Torch.FP32.Tensor` arguments and logical Tensor cells. Grid coverage
and program separation are proved separately. An arbitrary-order whole-grid
execution or interleaving theorem is not part of this bundle.

The selected Python is unchanged in this proof task. Its SHA256 is
`41d896ade9b5f5d6a0a3fdc3843f36124c6417d426f590ecb35c024f104f8b53`.
It retains the earlier vector-beta offset, scratch initialization,
canonical-stride and launch-device repairs. The original Python and all 64
pre-existing task files are preserved under the evidence directory's
`baseline/tree`, with their hashes in `baseline/manifest.json`.

## Entry domain and storage

`Contract.lean` exports the executable finite `kernel_check` and the proof that
it is equivalent to `KernelContract.InputContract`. Admission requires positive
B/H/T/K/V, ordinary contiguous FP32 CUDA tensors on one device, canonical dense
strides including singleton axes, no negative/conjugate views, disjoint argument
windows, and aligned pointers. The existing source grid, block and signed32
span bounds are retained: BH <= 65535, value-grid sizes < 2^31, and
BK*BBV <= 1048576. Native pointer windows include masked formed addresses and
the lower reserve for reverse retreats. This is metadata admission, with no
kernel-result or host-success premise.

BK is nextPowerOfTwo(K), FBV=min(nextPowerOfTwo(V),8),
BBV=min(nextPowerOfTwo(V),32), and BNV=ceil(V/BBV). Neither K nor V needs to
be a multiple of a tile. Source NK is 1. The raw kernel-entry shapes are:

| Roles | Shape |
| --- | --- |
| q, k | [B,H,T,K] |
| v, do | [B,H,T,V] |
| beta | [B,H,T] or [B,H,T,V] |
| h0, ht, dht, dh0 when enabled | [B,H,K,V] |
| o, dv | [1,B,H,T,V] |
| dq, dk | [BNV,B,H,T,K] |
| scalar dbeta scratch | [BNV,B,H,T] |
| vector dbeta scratch | [BNV,1,B,H,T,V] |

Forward overwrites v with residuals. Each backward program writes its owned
dq/dk scratch plane and dbeta slice, plus dv and optional dh0. Unowned vector
dbeta scratch stays unchanged; its host zero-initialization and later host
reductions are outside this kernel-only claim.

`KernelTensor.placement` constructs the finite source-write map from metadata
and proves its bounds and pairwise byte separation. No output coverage premise
is supplied by the caller. The conclusions retain Tensor identity/pointers,
all exterior bytes, and every untouched input role. `KernelIndex` connects
allocation-relative source indices to actual Tensor byte addresses and proves
active, masked formed, and reverse-retreat index bounds.

## Independent mathematical result

For each head/value column, with pristine x=v and supplied-or-zero H0:

```
r[t] = x[t] - dot(H[t], k[t])
u[t] = beta[t] * r[t]
H[t+1] = H[t] + k[t] * u[t]
o[t] = scale * dot(H[t+1], q[t])
```

`TensorBridge.kernel_forward_real` proves this K-dimensional recurrence for actual output,
residual, and optional terminal-state cells. `TensorBridge.kernel_backward_real` proves the
independent reverse derivative: dv, scalar/vector dbeta, dq/dk tile
contributions, and dh0. It includes the directional derivative of the
independent sequence loss with terminal cotangents. Key-padding lemmas prove
that zero-extension to BK preserves the Real state, output and gradient.

`kernel_saved_from_forward` derives backward's residual relation from actual
forward source stores, even when FBV differs from BBV. Each source entry has
its own program IDs and the same pristine memory; `kernel_forward_programs`
constructs this family from the complete source-execution theorem. Its saved-cell and
read-only-primal identities describe which forward cells are passed to
backward; no residual formula or backward output is assumed. The full tensor
VJP and scalar-beta sum identities are exported separately. The Real proof
interprets arithmetic over Real; it does not assert that rounded FP32 results
equal exact Real arithmetic.

The Real scale is the exact rational projection of the actual FP32 launch
word. Default scale uses the platform's Python binary64 `K ** -0.5` operation
and explicit scale denotes a positive finite Python float. Both use the same
selected object's runtime FP32 conversion. Positive values that underflow to
launch +0 remain admitted.

## Independent native result and compiler boundary

`TensorBridge.kernel_native_tensor` proves the complete source run, every expected native
word, Tensor readback and storage frames for both kernels. The independent
native recurrences preserve the padded reduction leaves, source operand
positions, explicit zero arithmetic, in-place residuals, reverse dk/dv stores,
barrier and scratch reloads. FP32 casts and memory boundaries materialize words
and prevent contraction across a store/reload.

`KernelCompiler.Binding` records the source/AST hashes, optional pointer slots,
integer specialization/divisibility, FP32 runtime scale slot, flags, tile
constants, and selected compiler options. Pointer attributes are checked
against the same Tensor arguments. Native add/sub/mul/FMA word functions,
binary64 scale conversion, contraction/operand routing and complete reduction
trees are explicit model data. Routing sees value-erased provenance. There is
no primitive-equals-software or whole-output numerical hypothesis.

The decoder is an abstract model port: no executable Lean decoder or complete
concrete routing instance for the compiled profiles is claimed. Twelve actual
installed Triton 3.7.1 binder/packing/offline SM90 compilations provide bounded
ABI and instruction evidence. Their TTIR/TTGIR/LLVM/PTX/cubin hashes, instruction
locations, reductions, shuffles and barriers are preserved in the compiler
inventory. This evidence does not establish universal compiler correctness or
ISA conformance. No GPU execution was performed and no numerical experiments
are deferred as proof obligations. See `compiler-boundary.md` in the evidence
directory for the exact limits.

## Validation and preserved host model

The evidence directory is
`.veritile-cache/kernel-evidence/fused_recurrent_delta/`.
`force_build.py` rebuilds the complete public task-local import closure using
the approved workspace v2 driver. `KernelAudit.lean` checks all public endpoints
against standard-base axioms and audits admission and independent references
for circular source/interpreter dependencies. `reproduce_inventory.py --check`
rechecks every compiled artifact and reproduces the saved inventory.
`audit_source.py --check` compares all 42 forward and 98 backward statement
nodes against the canonical Python AST after the documented syntax
normalizations and two scalar-beta pointer regroupings.

Existing CPU AST/storage replay is retained: 30 rejection controls, seven full
forward/backward admissions, 77 successful program executions, and one expected
out-of-bounds control under the historical singleton-stride guard. These are
bounded examples and do not replace the universal Lean proof.

The old wrapper/autograd development is available through `Misc/HostWhole.lean`,
`Misc/HostContract.lean`, `Misc/HostCorrect.lean`, and `Misc/HostFPEquiv.lean`; its earlier README
is preserved as `HostREADME.md`. It is separate from the kernel public bundle.
Host allocation, sum/squeeze/to operations, autograd return structure, device
launch completion and allocation/resource failures are outside this proof's
kernel-entry scope.

[IO.lean](IO.lean) declares the finite FP32 input/output windows, exact
strides, masks, in-place residual alias and complete per-program source
invocation. Separate mutually exclusive scalar/vector beta views share the
same original region. Backward scratch planes are explicit output channels:
all their source writes remain observed, including the intermediate dk/dv
writes whose rounded values are read again. The exact frame excludes only
the original writable cells; it does not permit writes across whole buffers.

[MathematicalIO.lean](MathematicalIO.lean) contains the independent
K-dimensional forward formula and full backward `Math.gradient` / initial
cotangent result. `SavedInputs` relates backward's saved residual to arbitrary
primal inputs only at times below T. The prior all-time saved-input relation
implies this finite relation. `Proofs/MathAgreement.lean` proves that changing
unused time coordinates preserves every relevant state, output and VJP;
`Proofs/IOSaved.lean` uses that fact to reuse the existing mathematical proof.
No zero-suffix constraint is imposed on the caller.

[NativeIO.lean](NativeIO.lean) selects the decoded object source, retains the
complete compiler binding and runtime UInt64 host scale, and observes output
words through the original instruction/routing model. Arbitrary input Nodes
remain admitted. `LogicalIO.lean` is a finite-input reference, with all padded
reduction lanes and every FP32 conversion retained. Only output payloads are
mapped to words; input encoding and numerical domains are unchanged. This is
a native source observation, with no second kernel or floating transformation
assumptions to report via `#print_fp_assumptions`.

Actual access safety is proved for the quotation's prefix, complete loops,
intermediate states, optional stores and backward recompute stage, then
connected to the shared IO bounds. Execution/output/frame proofs reuse the
existing complete source results. Former raw endpoints, coverage and geometry
aliases retain their declarations in `Misc/KernelCompatibility.lean`; the old
`Proofs/KernelCompatibility.lean` path is an optional compatibility import.
`Misc/TensorBridge.lean` is unchanged, including rational scale projection, byteheap
observation and input/physical frames; it uses the same checked execution
foundation. The format revision adds no numerical assumptions, experiments,
independent replay or GPU runs.

## Complete runtime program pair

`Runtime.forward` and `Runtime.backward` in `Kernels.lean` are the actual
original complete ordinary quotations. Both scalar/vector beta specializations
are displayed explicitly. The forward candidate `Runtime.forwardDirectOutput`
removes the zero key-grid term from the output address:

```text
(i_bh + i_k * B * H) * s_vo_h  →  i_bh * s_vo_h
```

The existing invocation fixes key program ID 1 to zero (source NK=1).
The candidate removes two scalar integer multiplications and an addition.
No runtime performance measurement is claimed. The backward candidate
transparently reuses its original complete source; both backward loops,
intermediate stores/reloads, barrier and all source write channels remain.
`FPEquiv.kernel_equivalent` selects either original stage and concludes both
actual successful executions, both source traces safe, every active typed
output equal, and both exact exterior frames through shared
`ProgramIO.Implementation` / `Equivalent`. The original per-program scope is
retained; no whole-grid ordering theorem is added.

The complete runtime quotations declare `tl.param scale : tl.float32`.
Their `Invocation.arguments` installs the selected model's FP32 launch value
in the real scalar carrier. For native interpretation that value is the
arbitrary original 32-bit scale word. Native instruction functions, reduction
plans and operand/FMA routing remain explicit model data, with all symbolic
input Nodes and all input words admitted. This pair makes no numerical
rewrite, uses an empty experimental rule set, and adds no hardware premise.

The prior literal-specialized reference is explicitly retained only in
`Proofs/LiteralSource.lean`. Its four ordinary quotations have full AST
`rfl` preservation checks against the former macro. `RuntimeBridge` proves
the actual runtime source/argument connection statement by statement and
through each loop successor. `RuntimeCorrect` transports the existing
independent Real recurrence and VJP to this actual runtime entry. The scale
marker is proof-reference data, not an actual source literal.

## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.

## Reader headline separation (2026-10-22)

The Real reader retains the two independent forward/backward kernel headlines.
The FP reader retains the actual runtime source pair selected by kernel kind.
Optional raw/native compatibility aliases are in `Misc/KernelCompatibility.lean`
and are no longer imported by `Correct.lean`. Checked completed-wrapper results
remain ordinary theorems in `Misc/HostCorrect.lean` and `Misc/HostFPEquiv.lean`,
with their original full names, statements and proofs. This changes entry roles
only; source programs, mathematical domains and numerical assumptions are unchanged.
