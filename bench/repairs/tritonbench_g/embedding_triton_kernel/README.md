# embedding_triton_kernel: checked FP32 kernels

For every active output row `r` and column `d`, the independent result is

```text
out[r,d] = initial_weight[initial_ids[r] - first,d]  if first <= initial_ids[r] < last
           positive FP32 zero                     otherwise
```

The required scope is the kernel. The wrapper supplies input assertions;
allocation, launch orchestration and returned-object proofs are optional.
`Correct.lean` proves complete per-program mathematical correctness.
`FPEquiv.lean` compares the original repaired kernel with the required
integer-index commuted variant: `pid * 64` becomes `64 * pid`.

`Kernels.lean` pastes both unchanged Python function bodies directly inside
`triton { ... }`, retaining every program/lane/ID int64 conversion and the
original variable names, masks, load options, loop and operation order.
The five annotated signed64 runtime arguments, four constexpr values and
FP32/int32 pointer metadata are supplied outside the bodies. The source still
uses 64 rows per program and one row per loop iteration.

There is no floating arithmetic. Valid gathers preserve payloads; invalid IDs
store positive zero. All FP32 words, including NaNs, infinities, subnormals and
signed zero, remain admitted. The original and existing candidate Python
functions are unchanged. No alternate handwritten execution AST is used.

The two Python JIT functions display the same difference. The existing wrapper
continues to select `embedding_kernel_repaired`; the candidate is
`embedding_kernel_index_commuted`. Both widen program IDs, lanes and loaded
IDs to int64. Invalid IDs are selected before subtraction, and masked output
rows use row zero before pointer formation. The host normalizes arbitrary
Python integer endpoints to an equivalent bounded interval for signed IDs.

## Input and memory contract

`Contract.lean` retains the exact Python admission conditions:

- IDs are contiguous signed int32 `[N]`; weight/output are contiguous FP32
  `[V,H]` and `[N,H]`, on the same CUDA device, with strided layout and resolved
  negative view flags. Contiguous offset views are admitted.
- Each tensor has fewer than `2^61` elements, `ceil(N/64) < 2^31`, and
  `H <= 2^20`. Nonempty data pointers are four-byte aligned.
- Output's accessed contiguous byte interval is separate from both inputs.
  The two read inputs may overlap. Empty intervals need no alignment.
- Endpoints are built-in Python integers. With `upper = min(last, 2^31)`,
  a nonempty intersection with signed int32 IDs and `H > 0` requires
  `upper - first <= V`. Empty, reversed or disjoint intervals zero-fill.

`checks_iff`, `argument_checks_iff` and the public
`TensorBridge.python_checks_iff_preconditions` establish the checked domain.
The same contract module contains the independent `target` formula.
`KernelContract` states the kernel's initial signed IDs, separate output role,
column coverage, nonoverlapping output rows, and active ID/weight/output
access bounds. `Bounds.kernel_contract` derives it for every program from the
checked Tensor metadata and signed ID storage. `KernelProof.accesses` relates
those bounds to the selected signed weight offset and masked output offset.
Masked rows and columns require no padded allocation.

## Public IO and independent formulas

`IO.lean` declares the actual source/candidate invocation and three typed
channels: 64 visited signed-int32 IDs, the FP32 vocabulary weight table, and
64 rows of FP32 outputs. ID windows use `pid*64+a`; weight-table windows use
`vocabulary_row*weightStride+d`; output windows use
`(pid*64+a)*outputStride+d`. The weight mask names only vocabulary rows selected
by active IDs. Consequently, unvisited weight rows and masked output padding
require no backing allocation or initial-memory binding.

The public mathematical formula explicitly indexes the original logical weight
table at `ids[a]-first`, or returns zero for an invalid ID. It is independent of
source execution and does not take an already gathered payload as its input.
`Correct.embedding_kernel_correctness` applies the shared
`ProgramIO.ImplementsOn` judgment to this formula, including actual source
access safety, successful execution, every active output and the complete
exterior typed-cell frame. The public domain retains signed-int32 IDs actually
read by the invocation; Real payloads remain arbitrary. Output separation,
column coverage and nonoverlapping row strides remain explicit layout premises.

`Proofs/StructuralAccess.lean` proves metadata, gathered-weight and output bounds
for an actual loop-body execution. `Proofs/ProgramSafety.lean` follows the source's
complete 64 iterations through their actual successor states. Candidate safety
uses equality of the pure initial multiplication step and the identical remaining
source. `Proofs/IOImplements.lean` derives those access obligations from the shared
IO's metadata bindings and bounds, reuses the existing execution proof, and
connects stored cells to the independent formula. The reader entries have short
composition proofs and checked statement-surface allowlists.

## Equivalence and retained proofs

`FPEquiv.embedding_index_commuted_equiv` remains the standard source-to-source
FP32/int32 program equivalence: exact integer commutativity preserves the entire
execution state and any failure. `#print_fp_assumptions` reports `none`.
The auxiliary theorem `FPEquiv.embedding_kernel_successful_equivalence` in
`Misc/FPObservation.lean` adds safe, successful shared IO for either actual source and the independent
bit-preserving vocabulary lookup. Invalid IDs produce positive FP32 zero;
all FP32 words, including NaN payloads, are admitted. No floating transformation
law, finiteness restriction or deferred numerical experiment is needed.

The previous raw Real execution theorem remains in
`Proofs/KernelRealResult.lean`. `Proofs/SuccessfulEquivalence.lean` retains the
complete two-source common-memory result, exact output/frame conclusion and
source-identity helpers. These ordinary supporting results preserve the prior
scope while operational statements stay outside the reader entries.

`Misc/TensorBridge.lean` retains Python admission and checked Tensor metadata-to-kernel
contract results. The existing constructive execution, grid, finite-index and
Tensor/host helpers remain available. Former host public results remain unchanged
in `Proofs/HostReal.lean` and `Proofs/HostFP.lean`; they are not required imports of
the canonical kernel entries. The regional IO does not claim a new flat-pointer,
compiler or runtime implementation proof. Python, `Contract.lean` and historical
sources are unchanged. The execution proofs now operate directly on the raw
bodies. `Execution.prefix_properties`, `chunk_memory` and `program_correct`
construct the actual widened row-register states, prove every output and retain
the entire exterior frame. Mathlib integer cast identities connect those signed
row expressions to the same logical coordinates. This changes no floating
operation or conversion and adds no premise to the public mathematical domain.
`StructuralAccess` and `ProgramSafety` follow the actual widened address
expressions through all 64 iterations. The original/candidate complete state
and failure equality now commutes the actual integer multiplication.

The manifest and mutation checks pin both full function bodies and all five
signed64 ABI annotations. They reject removing any of the four widenings,
changing selection/subtraction order, or replacing an annotated argument with
an unbounded scalar declaration. Native bounds, physical storage, mathematical
formulas, input contracts and all existing public correctness/equivalence scopes
are retained. No new numerical law or GPU experiment is introduced.

## Original failures and validation

The focused checker retains concrete original failures: strided ID and feature
views are read as flat storage; overlapping weight/output changes later reads;
a masked int32 default `2^32` wraps into a valid vocabulary ID; and an int32
row-stride product can overflow. The explicit guards and widened repaired
source address these cases within the stated domain.

Targeted Lean builds check both public entries and their helper closure, with
standard-base axiom and independent-target audits. `check_contract.py` replays
actual Python kernel ASTs with a small CPU memory model and mocked CUDA launch
metadata, covering guards, empty/tail cases, offset views and original defects.
`check_kernel_equivalence.py` additionally executes both complete JIT bodies
in Triton's actual CPU interpreter: 10 cases, 20 launches, including a 65-row
tail, padded widths, extreme signed IDs, an empty vocabulary, offset storage,
exact special-value bits, input frames and exterior output frames. It also
checks that the candidate Python AST differs only in the first product.
These checks do not execute GPU hardware.

`Correct.lean` and `FPEquiv.lean` retain the main gather/zero and source-equivalence certificates. Optional Python admission and checked Tensor metadata-to-kernel contract helpers are in `Misc/TensorBridge.lean`, under `EmbeddingTritonKernel.TensorBridge`.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/FPObservation.lean](Misc/FPObservation.lean). Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.
