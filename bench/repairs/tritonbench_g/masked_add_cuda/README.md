# masked_add_cuda

`Correct.lean` and `FPEquiv.lean` retain the kernel specifications. The optional
Python-check correspondence is `TensorBridge.python_checks_iff_preconditions`
in [Misc/TensorBridge.lean](Misc/TensorBridge.lean). Neither primary entry imports it.

The guarded Python source preserves the original in-place kernel and exposes
its input contract. `Correct.lean` proves kernels only. `FPEquiv.lean` compares
the original with the existing addition-order candidate; neither public entry
models wrapper execution, allocation or returned Tensor objects.

## Contract and independent formula

- Grad and data are strided CUDA FP32 tensors. The mask is signed int32 or
  int64. Shapes and devices agree; all three inputs are contiguous and have
  no unresolved negative view flag.
- `numel < 2**31`. Nonempty float pointers are aligned to four bytes; mask
  pointers are aligned to four/eight bytes. Empty inputs query no pointers.
- Grad's live byte interval is disjoint from both readonly inputs. Data and
  mask may overlap each other, including a mixed-dtype view of the same bytes.
  Disjoint windows of one allocation and contiguous storage-offset views work.
- Alpha is a built-in Python float, or the original default integer zero.
  The wrapper passes `float(alpha)` as a runtime FP32 scalar. NaNs, infinities,
  signed zeros and nonbinary masks are admitted.
- The existing launch uses block size 1024 and `enable_fp_fusion=True`.
  It mutates the original grad directly, makes no copies, and returns `None`.

`Contract.lean` defines the independent Real formula:

```text
grad_after[i] = grad_before[i] + data_before[i] * alpha  if mask_before[i] == 0
                grad_before[i]                        otherwise
```

Mask conversion tests nonzero; it is neither low-bit truncation nor the
original docstring's `(1-mask)` formula on arbitrary integers. For mask `2`,
grad `10`, data `2`, alpha `.5`, the kernel preserves `10`; `(1-mask)` gives `9`.
Real semantics specify the mathematical target, not exact FP32 arithmetic.

## Source and proof entries

| File or entry | Responsibility |
| --- | --- |
| `Kernels.lean` | One readable typed Triton DSL macro, original `grad += p*alpha`, and actual candidate `grad = p*alpha+grad`; signed mask widths remain explicit. |
| `Contract.lean` | Independent formula and the exact Python admission domain. |
| `IO.lean` | Public typed grad/data/signed-mask channels, source invocation and native scalar ABI, explicit read windows and exact veto-dependent store footprint. |
| `Correct.masked_add_kernel_correctness` | Successful original kernel execution, formula for every in-bounds lane, and full frame outside the active write set. |
| `KernelRealResults.masked_add_commuted_correctness` | The same Real result and frame for the actual candidate. |
| `KernelRealResults.masked_add_access_safe` | Each input/output access is within its stated regional bounds, without hidden readonly-input separation. |
| `TensorBridge.python_checks_iff_preconditions` | Ordered Python-model checks accept exactly the declared input contract. |
| `MemoryContract.checked_tensor_reads` | Shared FP32 Tensor laws and separate integer-width laws connect the checked actual pointers to logical input values using one byte reader. |
| `FPEquiv.masked_add_commuted_equiv` | Both complete source kernels succeed and agree on every typed memory cell under the selected fused graph. No numerical assumptions. |
| `KernelNativeResults.masked_add_kernel_result` in `Misc/NativeCorrect.lean` | Active FP32 result is one FMA in alpha/data/grad order; vetoed words and the complete exterior memory frame are preserved. |

The two reader entries each contain one public specification. `Correct` uses
`UKernelIO.RegionalImplements`; `FPEquiv` displays the actual source pair as
`original ≡[[]] commuted` and reuses the complete typed-memory proof.
`Misc/NativeCorrect.lean` retains the shared actual-schedule `ProgramIO.EquivalentOn`
theorem, including declared bounds, actual trace safety, active outputs and the
exact exterior frame.
The independent Real result on vetoed lanes follows from the input binding and
exact frame; those lanes are not falsely declared writable.

`Proofs/KernelRealResults.lean` retains the raw state and candidate-Real
results as ordinary theorems. `Proofs/KernelNativeResults.lean` contains the
main pair proof; its independent single-source native-word result and dedicated
`reference` are in `Misc/NativeCorrect.lean`, with their full names preserved. `Proofs/IOReal.lean` and `Proofs/IOEquiv.lean` compose the existing
execution foundations with the shared IO; `Proofs/AccessSafety.lean` checks both
actual source traces. These supporting modules import no public reader entry.

The kernel theorems quantify independent logical inputs and bind them to entry
memory. They do not require a pairwise-disjoint flat allocation model. Raw-word
leaves in the native graph represent every FP32 word. The equivalence domain
requires leaves only for source-active grad operands; arbitrary data expressions
and vetoed grad expressions remain admitted, exactly as in the retained raw proof. `MemoryContract` connects flags to actual
contiguous storage and logical reads; `Binding` retains concrete byte/store
lemmas and the finite-pointer correspondence.

Existing `Grid`, `Schedule`, `Host`, `Binding`, and `Words` helpers remain
available for optional use. Whole-wrapper proofs are not public acceptance
requirements. The old scalar/FMA-to-IEEE predicates in `Words` are optional
primitive-model comparisons, not optimization assumptions or experiment TODOs.

## Floating-point graph and compiler observation

The source's multiply/add is fused in the selected native compilation.
`Graph.nativeAlgebra` stages the product and consumes either addition order as
`fma(alpha,data,grad)`. `Variant` proves successful execution of both distinct
canonical DSL sources and equality of their complete memory results. It does
not assume split multiplication/addition equals an FMA, or use a split
ADD-COMMUTE numerical law. The FMA implementation and scalar ABI are primitive
modeling boundaries; there is no task numerical experiment obligation.

`Proofs/check_lowering.py` extracts the actual Python kernel and mechanically
constructs only the existing addition-order candidate. Offline compilation
with Triton 3.7.1, SM80/SM90, int32/int64 mask, runtime FP32 alpha and fusion
enabled gives identical executable PTX instruction streams for all four pairs.
Comparison removes source locations, declarations and function parameter-name
differences; it preserves every executable instruction and operand. Each
thread emits eight `fma.rn.f32` operations in alpha/data/grad order, with no
split float multiply/add. This is a scoped compiler observation, not GPU
execution or a proof of the compiler. Artifacts and hashes:
`/tmp/veritile-masked-add-candidate-lowering/report.json`.

## Targeted checks and retained defects

```bash
python bench/repairs/tritonbench_g/masked_add_cuda/check_contract.py
python bench/repairs/tritonbench_g/masked_add_cuda/Proofs/check_lowering.py
```

The contract check passes 27 admissions, 42 rejections and seven retained
original counterexamples. It interprets the actual unchanged Python kernel
with real CPU backing storage and mocked CUDA metadata. This covers empty,
scalar and tail shapes; storage offsets; singleton strides; extreme/nonbinary
signed masks; readonly aliases; exceptional-value veto preservation; and
rejections of invalid contracts. It uses simple exactly representable CPU
arithmetic and is not a floating-point equivalence experiment.

The retained counterexamples are stride-two addressing, unequal shapes,
shifted grad/data aliasing across programs, grad/mask aliasing changing a later
veto, lazy negative views, the nonbinary `(1-mask)` formula, and signed-int32
offset overflow beyond the admitted extent. At `n=2**31+1`, `pid=2**21`, lane
zero wraps to `-2**31`; the runtime bound is int64 there. The global runtime
grid-product/segmented-launch compatibility work remains deferred.

GPU execution and independent replay have not run. The targeted Lean build
status and exact public-file hashes are recorded in the task delivery packet.

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean), [NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.

Source readability cleanup (2026-10-21): `Kernels.lean` now displays all four complete ordinary `triton` quotations for the original/candidate and int32/int64 mask branches. `Proofs/LegacySurface.lean` archives the previous macro source; `Proofs/SourcePreservation.lean` proves equality of both complete source functions by mask-storage cases and `rfl`, for every region, extent, alpha and block size. Public declarations, the fused model, input domain and existing proof completion status are unchanged.
