# FP32 RMSNorm llama kernel proof

The selected Python kernel and guards are unchanged. The admitted profile uses
FP32 input, weight and output storage, a block of 16384 lanes, eight warps, and
the source's FP32 → FP16 → FP32 output round trip. Both source loops execute once
for the checked feature width, `1 ≤ N ≤ 16384`.

`Kernels.lean` contains one authoritative `llama_kernel%` quotation. The erased
Real interpretation binds its epsilon leaf to the mathematical host value; the
native interpretation reads the actual FP32 runtime scalar register. The local
frontend only seeds that external register's type and emits no placeholder
statement. `source_eq` links the literal binding definitionally to the original
typed source. Both source and candidate preserve the two loops, casts, masks and
stores; the candidate commutes one natural-number pointer multiplication.

`Correct.rmsnorm_kernel_real` uses the shared
`UKernelIO.RegionalScheduledImplements` judgment. `IO.lean` exposes the complete
matrix/weight/output windows and both canonical source variants. The theorem
retains arbitrary Real row and column counts, including empty dimensions and
multiple source blocks; it is not restricted to the checked host profile.
`RealPublicIO` composes the existing arbitrary-state row execution theorem and
complete two-loop safety proof over all rows. The shared launch preserves local
registers; the row theorem already permits arbitrary registers. The original
clear-register full-grid result is retained unchanged in `Proofs/RawCorrect`.

`FPEquiv.rmsnorm_kernel_equiv` states `original ≡[[]] indexCommuted` for the
two existing complete typed source grids with runtime epsilon. The prior
`ProgramIO.Observes` native word formula and `ObservedEquivalent` full-grid
result live in `NativeCorrect.rmsnorm_kernel_native`, with the ordinary
`FPEquiv.rmsnorm_kernel_native` compatibility alias retained in `Misc/Compatibility.lean`. `NativeIO.lean` binds the actual runtime epsilon register and the original
clear-register row setup. Its native domain remains `0 < N ≤ 16384`, with
arbitrary symbolic input terms. `NativeSafety` proves both actual typed source
traces, including the masked input/weight loads and output stores. Both shared
judgments include successful execution, every typed output and the exact frame.
Neither main statement exposes an initial state, runner or existential memory.

The optional complete Tensor applications are `TensorBridge.kernel_correct`
and `TensorBridge.kernel_native_equivalence`. Their statements are unchanged:
they expose complete source/candidate executions, all Tensor values, equality
of output memory with an independent reference update and the exterior frame.
The native theorem additionally gives every output byte and preservation of
the entire original input/weight backing allocations. The unrestricted raw
source/candidate equality, including failure, remains in `Proofs/RawNative`.

`KernelAPI.Binding` contains actual `Torch.FP32.Tensor` values, their ordinary API
laws, checked metadata, and fresh `empty_like` allocation facts. Logical input
words and physical source addresses are derived from those tensors. Pointer
bounds include each active four-byte span; signed-i32 arithmetic covers every
padded lane. Input alignment follows the existing nonempty guard. Input/weight
aliasing, storage offsets, singleton strides and empty leading dimensions remain
admitted. No execution result or numerical formula is supplied by the binding.

`Contract.python_checks_iff` exactly matches the executable guard list. It requires
contiguous strided FP32 tensors with no lazy negative/conjugate bits, X rank at
least one, W rank one, the same CUDA device, W.numel() = N,
`X.numel()+16383 < 2^31`, and a finite positive Python float epsilon. Nonempty input
and weight data pointers must be divisible by 16. There is no finite-data or
finite-converted-epsilon restriction. `EpsilonABI.convert` decodes the actual
binary64 host word and rounds once to FP32 with ties to even; positive underflow
to zero and overflow to infinity are covered explicitly.

The independent mathematical target is the conventional row RMSNorm formula.
The independent native target retains ordered casts, zero padding, initial
accumulation, reduction, division, square root, weight multiplication and the
FP16 output round trip. Reduction plans are explicit scalar trees with every
source lane exactly once. Compiler choices describe only structure and scalar
lowering; they cannot supply a tensor, reduction or kernel result. See
`NUMERICS.md` for their exact scope. The Real theorem does not equate rounded
native arithmetic with real arithmetic.

`Misc/Reader.lean` prints and audits the current public statements. Earlier host
models remain compatibility artifacts; the current public proof is kernel-only.
No core semantics, Python guard, Lake target or aggregate ledger is changed here.

Run `python bench/repairs/tritonbench_g/rmsnorm_fused_llama/check_contract.py` for
the existing source-pin and wrapper controls. They compare kernel/decorator ASTs,
match guards, and exercise 31 rejection cases and 15 direct-view cases, including
empty unaligned inputs, aliases, exceptional words and launch failures. These
controls use CPU arithmetic and mocked CUDA launch/context, and do not establish
GPU execution or native hardware conformance.

`TensorBridge.python_checks_iff` remains the optional check-correspondence
theorem. Existing `KernelSource`/`RuntimeSource` bindings reuse the same
`RealGrid.complete` and `RuntimeGrid.complete/graph_result` foundations and
prove the coordinate correspondence. `IO.lean` adapts the independent Real
formula in `Proofs/RealGrid.lean` to finite logical channels. The optional native
adapters and their complete observation theorem are in `Misc/NativeReference.lean`
and `Misc/NativePublicIO.lean`; the main execution/equivalence helpers remain in
`Proofs/NativePublicIO.lean`. Read-only X/W aliases and the existing output
separation conditions are retained.

The IO-format change is checked by targeted Lean builds, including both public
entries, preserved raw helpers, unchanged Tensor bridge and relocated reader.
It adds no numerical assumption and runs no GPU or native-conformance experiment.

## Selected pair review (2026-10-21)

`PairIO.lean` uses the shared `ProgramIO.Implementation` view to expose both
complete actual row schedules: the retained runtime-epsilon source and its
existing integer output-pointer product candidate. `Proofs/PairEquivalence.lean`
directly lifts the existing full-grid certificate. Both loops, FP32 accumulation
and reduction order, arbitrary original symbolic inputs, runtime FP32 epsilon
word, native reduction model, FP16 output round trip, successful safe execution,
all output cells and exterior frames are retained.
`#print_fp_assumptions rmsnorm_kernel_equiv` reports no experimental rule. The
original `0 < columns ≤ 16384` domain and separation premises remain in the main
pair. `Misc/NativeCorrect.lean` retains the independent native formula with its
observers, and independent Real correctness retains arbitrary/empty dimensions.

Optional native observations, Tensor/host applications and reader/audit entries
are in `Misc/`. Import these modules explicitly when needed; the primary
`Correct.lean` and `FPEquiv.lean` entries do not re-export optional results.
Legacy native compatibility names are available from [Misc/Compatibility.lean](Misc/Compatibility.lean).

## Source-only cleanup (2026-10-21)

`Kernels.lean` now gives complete ordinary quotations for `kernel`,
`runtimeKernel`, `indexCommutedKernel` and `runtimeCandidate`. The runtime
forms declare `tl.param eps_arg : tl.float32`, with no dummy scalar or custom
frontend. The original and existing candidate retain both loops, all masks,
reduction order, FP32 operations and the FP16-to-FP32 output round trip.

`Proofs/SourcePreservation.lean` proves complete function equality by `rfl` for
all four programs against the archived prior frontend/macro source. The generic
`indexCommuted` AST transformation is retained only as a compatibility proof
helper in `Proofs/IndexTransform.lean`; the authoritative candidate quotations
do not use it. Public declaration names, original domains/scopes, runtime
binding, primary equivalence and numerical dependencies are unchanged. Existing
proof completion counts/status are retained.
