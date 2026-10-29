# Liger layer normalization kernels

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/KernelAudit.lean`, `Misc/KernelProofAudit.lean`, `Misc/HostCorrect.lean`, `Misc/HostFPEquiv.lean`.

Current source-pair completion (2026-10-21): `FPEquiv.lean` proves two actual
related program pairs with an empty experimental rule set. The forward candidate
shares the identical FP32 `X_row - mean` vector between variance and affine
output instead of computing it three times. The backward candidate converts the
column-count denominator once before the complete row loop instead of twice
per executed row. Both preserve the exact ordered numerical graph, all original
outputs, source access safety and exterior memory frames. Empty backward groups
still write zero DW/DB partials. No measured speedup is claimed; compilers may
already perform these common-subexpression and loop-invariant optimizations.
The complete candidate source is in `Kernels.lean` and its Python companion is
`layer_norm_liger_optimized.py`. Optional original-source native observations
remain ordinary theorems in `Misc/NativeCorrect.lean`.

The selected Python source is `layer_norm_liger_repaired.py`, SHA-256
`6ba82f8c92986e02bca37469e5c17f2e1471137f939191d7fef4b48bd7266a28`.
The selected original Python and both original Lean source functions remain
unchanged; the additional candidate changes no numerical operation or reduction
order and needs no numerical experiment assumption.

`Kernels.lean` contains readable typed forward/backward source quotations and
parameters. Splitting/execution helpers live in `Proofs/`. `Contract.lean`,
`Correct.lean`, and `FPEquiv.lean` describe the actual kernels. `IO.lean`
exposes typed channels, regions, masks, row strides, scalar statistics and the
exact executed backward row interval. `Correct.lean` uses shared `ProgramIO`
judgments with independent Real targets; `Misc/NativeCorrect.lean` retains
the corresponding ordinary native theorems. The previous host contract and public results are retained as
`Proofs/HostContract.lean`, `Misc/HostCorrect.lean`, and
`Misc/HostFPEquiv.lean`. Their historical explanation is in
`Proofs/HOST_HISTORY.md`; host preparation, launch delivery, allocation, autograd
return packaging, and the Torch sum after backward are separate from this
revision's kernel claims.

The public contract binds actual FP32 Tensor descriptors, logical indexing,
storage identity/offset, capacity, data pointer, actual `stride(0)`, dtype, device,
and view flags.
It concerns already prepared contiguous arguments. It performs no input copy.
X/W/B and backward dY may share read-only physical storage, including different
views with nonzero offsets. Alignment is required only at the checked nonempty
data pointer; there is no new backing-storage alignment assumption and no
finite-element restriction. The block is the source's next power of two, at
most 65536. Backward uses the source's exact ceiling-divided row groups.

Every pointer reads from one physical heap. `KernelBoundary.logicalEntry_eq`
proves that the independent logical Tensor input overlay equals that physical
entry state. Both retained Real and FP `TensorBridge` theorems use this binding; it is
not unused metadata. `commit` realizes exactly the source write set at the bound byte
addresses. Its readback theorem uses explicit output-address separation. The
results include source-region frames, full physical-memory frames and
preservation of all backward read-only Tensors. Kernel parameters retain the
actual observed input strides. `RowStride` is the ordinary Tensor row-index API
law. Output stride identities come from the actual `torch.empty` allocations;
no input-stride guard is added. Contiguous singleton rows may have zero or huge
strides: they are not replaced with the column count.

The DSL uses natural index carriers. `KernelStrides` models the actual prelude
product and every loop-carried pointer increment modulo 64 bits. It proves that
every active native address equals the logical Tensor address and composes
these facts into both retained Tensor Real and FP theorems. Empty groups have no input
reads; the complete source reference then contains only the zero-partial
stores. After-last pointer increments are unobserved. The native input stride
products need not be overflow-free: rows=1, programs=3, pid=2, stride=2^62
produces signed64 -2^63 in the idle prelude, while the canonical-stride product
would be 2. The new theorem explicitly proves this example and does not equate
those unused values. `NativeIndex.Safe` concerns control/grouping and canonical
allocation offsets. Active pointer range/alignment follows from `valid_access`.

The independent Real forward target is population mean/variance, reciprocal
square root of variance plus epsilon, and affine normalized Y. Every Mean,
RSTD and Y cell written by the selected row is included. The backward target
uses the actual saved Mean/RSTD Tensor values, computes every DX cell in the
program's row interval, and includes both complete DW/DB partial rows. Empty
program groups write zero partials. The theorem is quantified over every
admitted program ID; kernel scheduling and the host's final partial-gradient
sum remain outside this entrypoint contract.

The retained Tensor FP result proves successful execution and whole physical-memory equality
to a separately defined ordered scalar graph over actual Tensor logical
values. It retains FP32 input/output words, zero padding, literal/epsilon/count
conversion, casts, subtraction/multiplication/division, rsqrt, complete
reduction trees, source row order, and selected local FMA contractions. The
compiler's primitive bindings, contraction choices and complete trees are
explicit model parameters. They are modeling boundaries, not numerical
experiment assumptions. No reassociation identity, separately-rounded/FMA
bit equality, FP-to-Real accuracy, or GPU execution is asserted.

Public results:

- `Correctness.layer_norm_forward_kernel_real`
- `Correctness.layer_norm_backward_kernel_real`
- `Kernel.FPEquiv.forward_pair` (`forwardOriginal ≡[[]] forwardSharedCenter`)
- `Kernel.FPEquiv.backward_pair` (`backwardOriginal ≡[[]] backwardHoistedDenominator`)

The previous `Equivalence.layer_norm_forward_kernel_fp32` and
`Equivalence.layer_norm_backward_kernel_fp32` remain auxiliary native results.

Validation: scoped Lean builds and axiom/noncircularity audits cover the public
results. The retained `check_source.py` passes ten actual-source CPU
forward/autograd/backward cases, fifteen guard rejections and the exact original
padding counterexample. `check_kernel.py` executes both actual kernel ASTs in
four offset/shared-storage cases, checking all six output buffers, empty
groups, one store per active cell, guards around every output, and input
preservation. It also executes the actual source on contiguous `(1,1)` views
whose retained strides are 0, 2^62 and 2^63-1 with three backward programs,
including the signed64-wrapping idle prelude. The CPU adapters replace CUDA
launch and Triton primitives; these
checks are not a GPU numerical proof.

The original unmodified benchmark's padding defect remains historical:
X=[2.25,3,3.75], W=1, B=0, epsilon=0.625 produced [-0.375,0,0.375] instead
of [-0.75,0,0.75]. The selected source already masks variance and backward
padding. That prior repair and its host acceptance are not reclassified as a
new kernel numerical experiment. This revision adds no deferred numerical
transformation experiment; native compiler/device conformance is an explicit
ordinary modeling boundary.

The initial kernel delivery's blanket stride-normalization claim was rejected
by independent review. The current correction binds actual strides and proves
observational equivalence, without changing Python or restricting singleton
inputs. Historical host files remain preserved artifacts; their old stride
normalization is not used to justify the corrected kernel entrypoints.

The main entries quantify `Forward.Parameters` or `Backward.Parameters` and
an arbitrary program ID, with logical arrays supplied by shared IO. They expose
no raw interpreter state or post-state existential. `Correct.lean` uses the
independent population-normalization/DX/partial-gradient formulas and exactly
the prior width, output-separation and DX-stride premises. The actual forward
and complete backward sources are proved trace-safe from these declared IO
windows, including the unmasked scalar statistics accesses and empty groups.

`Misc/NativeCorrect.lean` retains the wider original domain: arbitrary region aliases,
strides, symbolic input expressions, primitive bindings, contraction choices and
complete reduction trees. Each of its two ordinary source-correctness theorems
includes exact symbolic `Implements` and native `Observes`, so the former exact
symbolic result is retained along with evaluated FP32 words. The independent
`Proofs/OrderedReference.lean` uses scalar equations and pure ordered writeback;
it contains no interpreter state or execution. Forward writes Mean then RSTD
then Y; backward permits earlier DX writes to affect later input reads and
finishes with DW then DB. This avoids adding Real separation premises to the
native theorem. Overlapping active logical inputs are bound simultaneously by
the shared IO relation, and source safety is a proved conclusion.

The four `ForwardCall`/`BackwardCall` TensorBridge results retain their complete
original statements and proofs, including physical byte frames and modular
native-index facts. They reuse the same execution/value/frame foundation.
Public readers do not import TensorBridge, and supporting Proofs do not import
public readers. `Misc/KernelAudit.lean` is the public/Tensor/retained-host audit entry;
`Misc/KernelProofAudit.lean` audits only the supporting foundation. This format
revision reruns scoped Lean checks only; historical CPU checks above are not
claimed as newly rerun, and independent replay/GPU status is unchanged.
