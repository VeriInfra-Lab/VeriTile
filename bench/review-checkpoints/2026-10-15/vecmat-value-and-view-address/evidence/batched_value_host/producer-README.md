# Batched vecmat original fp32 values and logical host/grid

This package adds eight Lean modules in a temporary overlay. It proves the
complete existing `TypedSource.original` kernel on fp32 words for arbitrary
natural model sizes, block sizes, program IDs and total typed initial memory,
then composes every program in the original wrapper's two-dimensional grid.
The prior `TypedExecution` fixed-size execution witness and algebraic
commutation theorem are not used as a value proof.

The public specifications are:

- `VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs.batched_vecmat_fp32_machine_correctness`
- `VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs.Host.batched_vecmat_logical_wrapper_correctness`

The first has no proposition premise: it proves actual source execution exists,
exact final whole-memory equality and a frame. Inputs include `FP.Config` and a
valid explicit `ReductionPlan` at every reduction length. Used numerical
callbacks are concretely bound to `FP.mul`, `FP.add`, fp32 positive zero and the
supplied trees. A separate theorem proves final memory is independent of all
unused callbacks. The legacy DSL `.real` channel holds `FP.Value .fp32` here;
there is no real arithmetic or identity-rounding interpretation.

`Proofs/ValueModel.lean` defines the independent scalar reference. Each source
iteration loads its BM×BK and BN×BM×BK regions from program-entry memory,
multiplies in fp32, reduces axis 2 using the explicit tree, transposes into BM×BN,
and adds to the previous fp32 carry. The reference preserves initial +0, all
tree seeds, every product, every rounded add and block order. It then performs
the source's ordered output-lane writes. All loads occur before that final
store, so arbitrary aliases are handled in this logical per-program model.

`Proofs/ValueExecution.lean` proves the source prologue and arbitrary loop step
execute successfully and preserve the complete value/memory invariant.
`Proofs/ValueComplete.lean` inducts over the entire dynamic K-block loop and
proves the final actual store equals the independent memory recurrence. The
source is the existing full typed original, with successful checked algorithm
admission; no narrowed kernel copy is substituted.

`Proofs/HostGrid.lean` uses the actual grid `(M/BM,N/BN)`, pid0 fastest. Actual
programs receive fresh registers and the preceding actual memory. The reference
receives preceding reference memory, never an isolated-result merge. Full-grid
execution, coordinate coverage, whole-memory equality and frame are proved.
Both reference layers pass non-circularity checks against source execution.

`Proofs/HostPrepare.lean` records the original order: request random fp32 CUDA
A[M,K], request random fp32 CUDA B[M,N,K], request zero fp32 CUDA output[M,N],
then K remainder/assertion, M remainder/assertion and N remainder/assertion.
Every allocation can fail before the guards. Zero block sizes produce the
corresponding division error at the original stage. Successful preparation
certifies each returned region/memory, dimensions, positive blocks, all three
divisibility conditions, and unchanged num_warps/num_stages (defaults 4 and 1).
The allocator is an explicit non-numerical primitive boundary. No random
distribution law is needed: the kernel proof covers every returned word.

`Proofs/HostAccess.lean` proves all unmasked addresses for launched programs and
active loop indices lie within the declared A/B/output extents. Under source
divisibility it also proves every output row/column and every K coordinate is
covered. These are natural element-address bounds, not finite-width byte-pointer
or native allocation proofs.

`HostCorrect.lean` composes the three allocations, exact guards and complete
actual sequential grid. Its frame is relative to the memory after allocations,
as it must be. Errors return before kernel launch; no false claim is made that
failed or successful allocation has no effect.

The original Python wrapper AST was separately run with CUDA allocation requests
recorded then serviced by real CPU Torch tensors, and the kernel replaced by a
launch recorder. Twelve accepted cases check regular/default/tuning cases,
zero dimensions/K, non-power-of-two blocks, negative blocks and invalid tuning
values. Twelve error cases check all zero-block/divisibility stages, negative
dimension allocation errors, and allocation failures before all guards. These
are source control/metadata observations, not GPU output evidence.

Remaining structural boundaries keep `standard_accepted=false`:

- The host proof uses natural integer metadata. The source also accepts some
  negative block sizes and passes negative launch grids or invalid block values
  onward. Arbitrary Python values/types, negative dimensions and negative tuning
  values are outside this Lean domain; observed errors are recorded separately.
- Allocation implementation, tensor freshness and physical separation, contiguous
  fp32 storage, representation of `.real` typed cells, byte addressing and pointer
  ABI remain unbound. Allocation errors are modeled through a primitive callback.
- Fixed-width sizes, address arithmetic and intermediate overflow are unproved.
- Native arange/block/resource legality, zero-grid/zero-K compilation and launch
  admission, and accepted tuning parameters remain unproved. Non-power-of-two
  blocks pass the Python assertions. The total software interpreter's success
  for BM/BN/BK=0 is not claimed as source-wrapper success: the wrapper model rejects
  zero divisors after the allocations in the original order.
- The chosen sequential schedule is not a native GPU concurrency refinement.
  Original fresh allocations should separate inputs and output, but native
  freshness, race freedom and schedule independence are not established here.
- The exact source correspondence is to the existing admitted DSL transcription,
  with static Python comparison and wrapper probes; no verified Python compiler
  translation theorem is claimed.

Separately deferrable numerical bindings are the actual device FP.Config,
rounding/FTZ/special values, native reduction tree and explicit seeds, and compiler
fusion/reassociation behavior. No structural limitation above is a numerical
assumption. No old mathematical/commutation proof is relabeled as this new value
contract, and no native compilation or GPU execution is claimed.

All new source files and evidence are under this temporary package. Repository
sources, lake registration and prior metadata are untouched; checks emit only
ignored `.olean` cache files. `evidence/lean-checks.json` records exact commands,
observed exits and source/log hashes. Integration requires independent scope
review and parent approval of the frozen receipt.
