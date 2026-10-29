# Contiguous FP32 last-axis softmax kernel

The API remains `softmax(input, mask=None, dim=-1)`. Input may have any nonzero
rank. `dim` must be `-1` or the last nonnegative axis index. An optional additive
mask may have different outer dimensions: only its last-axis width and product
of outer dimensions must match the input. The returned Tensor preserves the
input's original shape.

Input and mask must be strided, contiguous FP32 CUDA tensors, on the same device,
with no unresolved negative-view bit. Nonempty tensors require an actual data
pointer divisible by four, a flattened row count below `2**31`, and a block
`max(next_power_of_2(width), 2) <= 2**20`. There is no additional total-element
limit. Empty axes impose no pointer or unused launch bound. FP32 input, mask,
output and computation are the selected task precision. NaNs, infinities and
subnormals remain admitted. `Contract.lean` expresses exactly this domain;
`TensorBridge.python_checks_iff_preconditions` proves admission in both directions.

The wrapper directly passes the original tensors with canonical contiguous row
width. It does not resolve, clone, view, flatten or reshape them. Empty shapes
return the fresh contiguous allocation without block setup or a launch. All
nonempty sizes use one valid row kernel launch; the original branch above
350000 rows passed nonexistent `n_rows` and `BLOCK_M` parameters. That source
bug is repaired, and large valid row counts are retained. The original shared
stride/layout defect is addressed by the checked contiguous domain and canonical
row stride. Storage offsets and irrelevant singleton strides remain supported.

The JIT change is limited to int64 row-stride/width arguments and widening
program/lane indices before multiplication. Its complete floating statements
are preserved: FP32 load cast, input maximum, subtraction, optional additive
mask, exponential, ordered sum, division and FP32 store. The mathematical Real
target is softmax of `input + mask`; the source deliberately does not include
the mask in the maximum. Large positive masks can therefore overflow native
exponentiation even when a differently ordered stable softmax would not. No
numerical rewrite hides that behavior.

`Kernels.lean` contains the two complete Python function bodies from
`softmax_triton3_repaired.py`, including both signed-int64 casts, the original
`mask_ptr is not None` test, explicit FP32 load casts, finite classifiers,
`1.0` literals, fast-path guard and division fallback. Pointer storage, signed
runtime scalar values, constexpr size and optional-pointer presence are supplied
outside `triton { ... }`. The caller binds presence to the existing `HAS_MASK`
argument; the actual optional-input interfaces supply `mask.isSome`.

The frontend preserves the conditional and rejects uses of the optional pointer
outside its non-None branch. `jit-correspondence.json` checks both raw bodies,
the dtype/scalar profile and the caller presence binding. No pointer address or
load is invented for an absent mask. The Python implementation, wrappers,
public mathematical/FP statements and input domain remain unchanged.

The historical manual AST omitted signed casts and specialized the mask test
in the body, so it is no longer used as an AST-equality certificate.
`Proofs/RealExecution.lean` proves actual execution, output values, masked access
safety and exterior frames for both mask branches. `Proofs/FP.lean`,
`PairExecution.lean` and `PairSafety.lean` independently follow the actual source
and the existing numerical model/rules. Launch selection and the unchanged
wrapper guards live in `Contract.lean`.

`Proofs/Binding.lean` interprets each formal pointer as a window starting at the
actual Tensor storage offset. For each row, it reads from the exact reached
physical state, executes the whole original source, and projects its active
stores back to the actual output allocation. The explicit memory-lens law is
constructed by record update in `Misc/TensorBridge.lean`; it is not an additional
runtime or numerical assumption. Read-only input and mask windows may overlap. Their source region identity is
derived from the actual Torch storage object; aliases cannot be assigned separate
backing memories.
`Proofs/KernelBinding.lean` supplies one physical-address reader for both input
and mask. Its `bound_load` theorem identifies the source cell with the checked
64-bit native pointer and then the original Tensor's logical coordinate. The
TensorBridge Real and FP results use this composition to obtain every source operand.
Output reads use the actual output Tensor's physical indices. The frame retains
every cell outside the active row's output write set, including other rows,
padding and cells outside the logical views. The output buffer's metadata and
shape are explicit kernel preconditions; its initial values are unconstrained.

The independent public entries are:

- `Correct.softmax_kernel_correct` and `softmax_masked_kernel_correct`: shared
  `MaskedKernelIO₁/₂.Implements` judgments for the two genuine optional-mask
  branches. The independent formula uses only active logical row values;
  source trace safety, flattened execution, outputs and exterior frame are proved.
- The ordinary `FPEquiv.softmax_kernel_fp32` theorem in `Misc/NativeCorrect.lean`:
  shared `ProgramIO.Implements` for the same
  source under the selected native FP32 model. It proves actual structural
  access safety, successful execution, every native output word at its selected
  column owner, and preservation of all exterior cells and their tags.
- `TensorBridge.python_checks_iff_preconditions`: exact two-way guard admission.
- `Proofs/Addresses.pointer_bounds` and `mask_exact`: signed finite-width index
  and mask correspondence, actual storage bounds and four-byte alignment for
  every input/mask/output access. The kernel uses the explicit output-buffer
  contract; the optional host proof derives it from allocator facts.

The kernel entries require neither a wrapper run nor allocation success. They
cover all valid program IDs of the existing row launch; empty tensors have no
active output coordinates, and the unchanged Python wrapper skips their launch.
The previous wrapper and returned-object results remain available separately
in `Misc/HostCorrect.lean` and `Misc/HostFPEquiv.lean`, under the `HostCorrect`
and `HostFPEquiv` namespaces. `Proofs/Host.lean` composes the rows and proves the
fresh returned allocation and caller frame when that additional host result is
needed.

The FP entry is parameterized by a V1/V2/V4 blocked-layout profile. Applying it
to a native invocation requires binding that parameter to the actual compiler
metadata; this remains an ordinary compiler boundary. It
retains the original 4/8/16 warp choices and exact register fold, simultaneous
butterfly and shared-memory exchange order. Exponentiation remains subtraction,
optional mask addition, multiplication by FP32 word `0x3fb8aa3b`, and `ex2.approx`;
normalization remains `div.full`. The profile is ordinary compiler metadata,
not an input restriction or a whole-row numerical law.

The six scalar raw32 semantics ports are `maxNum`, `subRN`, `mulRN`,
`exp2Approx`, `addRN` and `divFull`; mask addition uses the same scalar add port.
These ports and compiler profile selection are explicit modeling boundaries,
not deferred transformation experiments. This revision makes no numerical
transformation within the retained single-source native theorem: no max-shift
move, reciprocal rewrite, FMA change, or reduction reassociation is introduced
there. The new program pair uses the admitted reciprocal rule described below. Native primitive conformance, including rounding, FTZ and exceptional
values, is not established by these Lean results. Negative-infinity padding
and overflow mean finite input leaves do not imply finite intermediate operands.
The FP theorem covers all raw words under the specified model; the Real theorem
is a separate mathematical interpretation. No FP-to-Real equality is claimed.

Ordinary trusted boundaries remain Tensor metadata/storage semantics, scalar
word/byte encoding, compiler selection of the specified ordered graph, and
race-free GPU execution. Successful fresh contiguous allocation is a boundary
only for the optional host results. The proofs
do not implement Torch, the compiler or asynchronous launch failure semantics.
Original benchmark files and historical acceptance records are unchanged.

Validation uses the relevant Lean closure and standard-base axiom audits, with
no new axiom or `sorry`. The independent formula/source review retains the
original >350000-row invalid-keyword branch and shared-stride/layout defects;
the checked repair addresses them without further domain narrowing. Run
`python bench/repairs/tritonbench_g/softmax_triton3/check_contract.py` for source,
actual guard and host checks: 18 CPU Tensor admissions, 27 rejections, different
mask outer shapes, arbitrary ranks, offsets and overlapping input/mask storage,
singletons, empty out-of-storage views, alignment, warp branches, metadata limits,
350001+ rows and the preserved subtract-then-mask numerical order. Allocation
and CUDA launch are mocked. No GPU numerical experiment was performed.

`IO.lean` is the shared public descriptor file. Its records expose the actual
source, input/output buffers, row windows, active-column masks and native
column-owner observation. The optional native mask channel is inactive when
`mask=None`; it adds no allocation, input restriction or phantom read.
`Correct.lean` contains the independent mathematical specifications, and
`Misc/NativeCorrect.lean` preserves the ordinary native theorem and its short
composition proof. They import no TensorBridge. `FPEquiv.lean` declares the complete runtime-guarded reciprocal program pair;
optional native results require an explicit Misc import.
The native shared IO is a regional contract; the stronger Real flat-pointer
transport and the existing Tensor physical correspondence remain separate.

`Proofs/RealIO.lean` and `Proofs/NativeIO.lean` assemble the shared judgments.
`Proofs/StructuralSafety.lean` checks every actual Structural load/store address
through the selected source trace, without assuming source safety or a numerical
identity. `Proofs/NativeIOResult.lean` binds logical raw words and the selected
native output owner to the complete source. Earlier raw operational theorems
remain reusable in `Proofs/KernelRealResult.lean` and
`Proofs/KernelNativeResult.lean`. The exact earlier Tensor statements remain
`TensorBridge.softmax_kernel_correct` and `TensorBridge.softmax_kernel_fp32`,
using the same source/termination facts and physical-address correspondence.

The 2026-10-20 public IO-format pass changed no Python source, source quotation,
contract, primitive numerical assumption or GPU validation status. Targeted Lean
builds include the descriptors, supporting proofs, both reader entries and
TensorBridge; statement-surface and standard-base axiom checks run in the entries.

## Program pair, 2026-10-21

`FPEquiv.softmax_reciprocal_equiv` is the sole kernel main specification. It
compares the complete original `softmax_kernel` with the direct
`guarded_reciprocal_kernel` under exactly the original typed input, optional-mask
and output IO. The shared `ProgramIO.Equivalent` judgment concludes both actual
source safety traces, both successful runs, active complete typed output cell
equality and both exterior memory frames. Every original FP32 word and both
optional mask cases remain admitted. The main proof imports no `Misc` result.

The candidate Python wrapper is `softmax_reciprocal`. It copies the original
wrapper guards, allocations, empty handling, warp selection and complete row
launch. The common source prefix retains FP32 casts, maximum subtraction before
optional mask addition, exponentials, padded sum and all addresses/masks. When
all reached numerator lanes are finite and the reached denominator is finite
and at least one, normalization uses a common scalar reciprocal followed by
lane multiplication. Every other row uses the original division. This reduces
per-lane division count on the fast path; no measured speedup is claimed. The
denominator test prevents the obvious tiny-denominator reciprocal overflow
example, including for arbitrary additive masks, without restricting inputs.

`Proofs/PairExecution` checks both mask selections and every runtime guard case.
`Proofs/Pair` connects actual source operands to the local numerical rule and
proves whole typed-memory equality. `Proofs/PairSafety` proves original active
bounds suffice for both sources and proves the original complete exterior
frame. `Proofs/SourceIO` connects these facts directly to the full main judgment.
The runtime classifiers have explicit exact `Ports` bindings to FP32 `finitef`
and comparison, with finite/nonzero soundness. These are primitive/model
bindings, not caller finiteness guards or whole-program numerical assumptions.

The only numerical rewrite used is the already admitted FP32 `DIV-MUL-RCP`, on
finite reached numerators and a finite nonzero reached denominator. Existing
evidence is `experiments/floating_point/supplement/report/summary.json`, entry
`fp32/DIV-MUL-RCP`, snapshot
`9b01836a349a09e938f62e813c56a99aed2d0e39bf801fb5b050d57000b5604c`.
The rule remains an explicit conditional numerical assumption. Finite-sample
admission is not universal IEEE bit equality or a whole-kernel error bound.
There is no new numerical relation or deferred experiment. Optional
`Misc/CheckedPair.lean` retains the empty/every-row raw-memory schedule instance
and derives count width from the unchanged input contract; it is an ordinary
theorem outside the main dependency chain.

## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.
