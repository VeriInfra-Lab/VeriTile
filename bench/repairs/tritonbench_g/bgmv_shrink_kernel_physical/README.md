The current selected checked implementation of `bgmv_shrink_kernel` is
`bgmv_shrink_kernel_repaired.py` in this physical-storage variant. The earlier
`../bgmv_shrink_kernel/bgmv_shrink_kernel_repaired.py` remains preserved, SHA256
`7152c240f53ba0c35f44fc3967491bb31e02c60a0cbe235cafdcfacaa0c3018d`,
together with its diagnostic and historical proofs.

The candidate replaces output/read StorageImpl identity comparison with the
physical half-open intervals returned by `untyped_storage().data_ptr()` and
`untyped_storage().nbytes()`. Empty intervals never overlap. Only output/read
pairs are compared; read/read aliases remain admitted. This checks entire
backing storage, including storage outside a tensor's logical view.

The candidate also requires `256 * BLOCK_N <= 2**20`, where
`BLOCK_N = max(1, next_power_of_2(N))`, immediately before the existing launch
path. This rejects the known N=4097 frontend tile defect. It is a bound on the
actual source broadcast tile, not a promise of compiler or hardware resource
success for every admitted shape.

Both new checks occur after the original `B == 0 or N == 0` return. All other
guards remain. Consequently all previously admitted empty no-ops survive,
including B=0 with N=4097. Same-StorageImpl empty calls, formerly rejected by
the identity guard, now return None without reading index values, launching a
kernel, or writing tensor storage. This explicit relaxation means empty admission
is not identical to the old wrapper. For nonempty calls the new physical guard intentionally
rejects previously admitted overlapping DLPack-backed tensors.

The entire JIT function and its launch arguments are unchanged. X and W retain
matching FP16 or BF16 storage. Their product and reduction joins retain that
format before FP32 accumulator extension. Y remains caller-supplied FP32.
Each active cell has 64 distinct FP32 atomic updates, including K=0. The native
target carries the actual per-cell permutation and does not apply real
associativity, commute atomic events, or combine them into one reduction.
Finite Python scaling may round to FP32 zero or infinity; that domain is retained.
The native target reads the actual FP32 launch scalar word through a separate
register. Its sign bit is preserved for Python `-0.0`, independently of the
positive zero literals used for accumulator initialization and masked loads.
The Real target uses the corresponding mathematical scalar. Relating a Python
float descriptor to its actual FP32 launch word is an ordinary local host ABI
boundary, not a whole-result numerical assumption.

`Target.lean` declares independent dense Real and typed numerical targets.
`Kernels.lean` contains the complete original Python function body and explicit
scalar-left candidate. Eight exact body mappings cover FP16/BF16 storage and
i32/i64 metadata pointers. Call metadata remains outside `triton { ... }`;
no template assignment, statement removal, guard inversion or cast rewriting
is needed to paste the original function body. The signed int64 aranges, return,
split-K loop, masks, augmented scaling and atomic call remain as written.
`jit-correspondence.json` pins the sources and ABI bindings. The separate
`check_pair_source.py` validates the complete candidate Python module, permitting
only the final multiplication operand exchange and entry/call names.

`check_candidate.py` executes the actual host AST against CPU tensor backing
storage and a launch-recording CUDA metadata stub. It checks the physical
overlap witnesses, metadata aliases, adjacent spans, read/read aliasing, N=4096
and N=4097, K=0, and the documented empty behavior. It does not claim GPU
numerical execution. Existing offline compiler evidence applies to the unchanged
JIT body and unchanged launch arguments; old large-tile frontend acceptance is
not promoted to full PTX success.
Numerical and tensor results describe successful completed launch observations.
Any issued prefix has the proven destination frame; CUDA failure does not imply
transactional rollback or a valid completed numerical result.

Evidence is written to `.veritile-cache/bgmv-shrink-kernel-physical-evidence/`.
The variant's original delivery did not change the canonical benchmark ledger;
the current ledger selects this physical-storage source.

`Correct.kernel_real` and the ordinary `FPEquiv.kernel_native` theorem preserved
in `Misc/NativeCorrect.lean` use the shared `PerAddressProgramIO.ImplementsOnCells`
judgment. The native result is auxiliary; the strict program-pair result is
`Equivalence.scalar_left_strict`. `IO.lean` exposes the actual
complete source catalog, runtime scale, typed finite input windows, in-place Y,
and each output address's 64-event order. The selected weight window is sized
from the metadata itself; kernel-level metadata is not newly restricted by the
wrapper's adapter count. All sentinel outputs remain observable, and N=0 still
checks every source CTA even though there are no output lanes.

`Mathematical.lean` declares the independent dense Real formula and unchanged
typed native recurrence. `Proofs/PublicIO.lean` composes actual source safety,
successful execution, event coverage, output locality, exact typed traces,
output binding and frames. None of those program properties is an assumption.
The native result retains the existing scalar-tree/compiler boundary and makes
no native=Real or floating rewrite claim.

`Proofs/IOProjection.lean`, `IOOldRun.lean` and `IOConnection.lean` connect the
original runtime traces to the same source traces on the same input memory.
They preserve arbitrary exterior memory and undef values, the stronger sentinel
frame, and the original four-byte physical output observations. The former raw
entry statements are retained in `Proofs/RealKernelResult.lean` and
`NativeKernelResult.lean`; `Misc/TensorBridge.lean` retains all original theorem
statements, caller-Tensor results and physical byte/backing frames.

## Program-pair review, 2026-10-21

`FPEquiv.lean` proves the actual complete `original ≡[[]] scalarLeft` pair.
`Kernels.lean` contains ordinary direct quotations of both FP16 and BF16
sources, with runtime FP32 scaling explicit. The existing scalar-left candidate
exchanges only the final `accumulator * scaling` operands. The original source
is connected to its historical presentation by complete execution equality and
access-safety equivalence in `Proofs/SourcePreservation.lean`; the source trees
are not claimed equal. `Proofs/SourceControl.lean` proves the Boolean guard and
constant loop-stride connections. The archived generator is outside the main
proof dependency closure.

`Proofs/PairConnection.lean` proves complete state equality and access-safety
equivalence, including failed executions, from the local FP32 multiplication
exchange. `Proofs/PairIO.lean` lifts this to every batch × 64 CTA and each
address's independent atomic order. Both sides establish complete source
success, safety, coverage, locality, tagged output equality and exterior frames,
including empty output axes. FP16/BF16 products and reductions, the K loop,
output conversion and atomic primitive are identical on both sides.

The primary instance fixes just the changed FP32 multiplication to the existing
strict `FP.mul cfg` model and reuses `FP.ScalarOpsLaws.mul_comm` for all words and
configurations. Other typed local operations remain explicit and shared.
There is no finite-value guard or experimental numerical rule. This model
certificate makes no hardware-conformance claim. Historical single-source
native results retain their names and statements in `Misc/NativeCorrect.lean`.


## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.

## Literal function-body migration

The public Real and FP readers and their contracts are unchanged. The proofs
follow actual integer registers produced by `.to(tl.int64)` and prove their
nonnegative address parts from the existing call geometry. The runtime FP32
scale stays a supplied register. The quotation inserts Triton's implicit
conversion to the destination element type at `tl.atomic_add`; it uses the
same conversion discipline as `tl.store`. The generic Lean output-type domain
is retained, while the selected Python wrapper supplies FP32 Y.

The proof chain reconnects complete source execution and safety, every batch
and all 64 split programs, sentinel rows, each output address's atomic order,
the independent Real formula, typed FP observations and exterior frames.
The FP pair reuses its existing strict all-word FP32 multiplication certificate;
no experimental rule or arbitrary cast identity is introduced.

The pure source checker and mutation tests run without Torch. The original
host guard checks are retained but require Torch to execute; source validation
does not count as a host or GPU run. The final frozen cumulative replay and
its exact environment limitations are recorded in the review checkpoints.
