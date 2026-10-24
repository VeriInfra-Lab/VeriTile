# add_value: checked contiguous FP32 contract

The independent result is `out[i] = x[i] + 10` at every logical coordinate,
with the input shape. `add_value_repaired.py` writes `output = x + 10`
directly and removes the `value` kernel parameter and launch argument. Its
second kernel writes `y = 10; output = y + x`. The original benchmark Python
is retained separately and unchanged.

The required proof scope is the kernel, as clarified by the user on
2026-10-19. The wrapper supplies input assertions. Allocation, launch
orchestration and returned-Tensor identity are outside the required proof.
`Correct.lean` states kernel correctness; `FPEquiv.lean` compares the two
kernel versions.

## Supported input and source behavior

`x` must be a strided FP32 CUDA Tensor, contiguous in the default format,
without an unresolved negative view bit, and with `numel < 2**31`.
Nonempty inputs must have a data pointer divisible by four. Empty inputs
impose no pointer-alignment condition and do not query the pointer.
Scalar tensors, singleton dimensions with unusual strides, contiguous offset
views, NaN/Inf, signed zeros, and subnormals are admitted. Read-only input
storage may be shared with other tensors; the output allocation is fresh.

The eight Python rejections precede allocation. `Contract.lean` names the
same conditions, and `TensorBridge.python_checks_iff_preconditions` proves exact
dynamic admission equivalence, including rejection of non-Tensor arguments.
No input copy, normalization, flattening, or chunked launch is used.
The wrapper calls `empty_like`, returns that exact object for empty inputs,
and otherwise launches `puzzle1_kernel` once on the input device with
`BLOCK_SIZE=1024`. The addend is the kernel literal 10, not a runtime input.

FP32 is the selected production profile. The original benchmark annotates
its scalar as `tl.float32` and always passes 10; this revision specializes
that constant in the source. With FP32 tensor inputs, both displayed
additions and the load/store use FP32. `scalarWord = 0x41200000` is proved
to denote exactly 10. Other input dtypes are outside this checked revision.

## Original failures and scope decision

The original wrapper checked only CUDA placement after allocation. Its raw
linear loads do not implement arbitrary logical views. For backing storage
`[1,70,2,80,3,90]`, a stride2 input logically contains `[1,2,3]`, but the kernel
produces `[11,80,12]`. An unresolved negative view similarly reads positive
backing values. At `N = 2**31`, the annotated int32 extent becomes negative
and all masks are false. The existing original `NativeArithmetic` proof
retains that counterexample; the targeted Python check also exercises its
boundary arithmetic without allocating a huge tensor.

This revision follows the approved explicit-input-contract scope. It guards
these unsupported layouts and extents visibly; it does not certify the
unchanged original wrapper. The prior checkout's general copying/chunking
repair remains a separate historical result. Contiguity alone does not
guarantee pointer alignment: actual CPU FP32 views over byte-offset storage
give pointer remainders 1, 2, and 3. Those cases motivate the explicit guard.
Their CUDA construction/execution has not been tested.

## Proof entries and trusted boundary

- `Kernels.lean` contains one authoritative readable Triton DSL source with
  explicit FP32 load, addition and store annotations, its commuted variant,
  and the independent Real formula. It imports no proof module. The selected
  `originalKernel n` fixes block1024 and contains `x + 10`; the commuted source
  contains `y = 10; output = y + x`. Neither takes a `value` input.
  There is no separately handwritten source AST.
- `Correct.addValueIO` explicitly declares the shared literal-add10 kernel's
  input/output windows and tail mask. `Correct.puzzle1_kernel_correctness`
  states `Spec.Real (addValueIO n ⊨ fun xs i => xs i + 10)`: each program
  terminates over any disjoint flat allocation satisfying the active-lane
  bounds, computes those output lanes, and preserves other cells, with
  arbitrary initial output and register contents. This per-program claim does
  not state Python admission or whole-grid execution.
- `Correct.addValueIO.kernel` and `FPEquiv.original.kernel` both use exactly
  `originalKernel n`. `Proofs/Projection.lean` connects its mathematical
  projection to the existing kernel Real proof helpers. Projection is confined
  to Real semantics.
- `FPEquiv.original` and `commuted` package the shared source pair and its
  actual `additionSite`. `add_value_commuted_equiv` states
  `original n ≡[[]] commuted n`, applying strict FP32 addition commutation.
  Its finite-operand condition uses the actual mask after the original load
  prefix: only lanes with `mask = true` require finite operands. Values in
  masked-off lanes are unrestricted because their results are not stored.
  This is a rewrite-site condition, not a new Python guard.
- The rewrite theorem executes the source's explicit FP32 load and compute
  operations through the FP32 engine and `FP.ProfiledMemory` view. Its `.real` type is the opaque
  data carrier; the retained instruction tags select FP32 semantics, including
  `M.fp32Load`. `FPEquiv.original_source` is a literal kernel identity.
  `Proofs/Rewrite.lean` executes both displayed sources successfully and
  proves equality of their complete typed memories. It uses the shared
  `FP.ScalarArithmetic.add_comm_strict` theorem at the actual load prefix. The local
  constant assignment is handled by execution, with no extra numerical law.
- `Proofs/MaskedDomain.lean` checks independence from all unspecified inactive
  load values, the empty domain, a one-element tail's complete memory
  equivalence, and rejection of a nonfinite operand when its lane is active.
- `Proofs/Source.lean` checks the shared source's explicit FP32 load, add10 and
  store forms before lowering; untagged floating instructions are rejected.
  Its finite interpreter executes int32 offsets, signed masks, one load,
  add10, and masked stores. `FPEquiv.source_lowering` binds the public source
  to that instruction sequence. Arbitrary
  inactive load values cannot change output. The kernel IO retains active
  access bounds and the exterior frame; no padded-tail allocation is required.
- `Contract.lean` and `Proofs/Contract.lean` retain the input assertions, exact
  guard/admission correspondence and the signed-index/mask connection under
  the admitted extent and launch geometry. Shared `VeriTile/Torch` metadata
  and memory/value bindings remain available for interpreting Tensor inputs.

The kernel proof starts from the declared memory and program-ID conditions.
The flat-memory IO states its bounds, active lanes, output formula and frame.
The wrapper is the source of input assertions, not a second program to prove.
Compiler, Torch allocator and GPU runtime implementations are outside this
proof scope. The selected grid remains one-dimensional and at most `2**21`;
no shared launcher compatibility workaround is introduced.

Earlier whole-wrapper Real and returned-Tensor proofs remain in the existing
`Proofs/Real.lean`, `Proofs/FP.lean` and host helper files as optional results.
The public `Correct` and `FPEquiv` entries no longer import those local host
proofs or require them for acceptance. This preserves earlier work while
applying the requested kernel scope; historical cleanup remains separate.

## Floating-point transformation and validation

Strict FP32 addition commutation proves the change from `x + 10` to
`y = 10; output = y + x` at the displayed finite-operand site.
`ScalarArithmetic.add_comm_strict` connects the shared
`ScalarOpsLaws.add_comm` proof to the fixed `Guarded.fp32Theory`, whose
arithmetic profile is `Software.strictFP32` with `.ieeeValue`. The public
theorem uses `≡[[]]` and has no empirical `R` parameter.

The historical `ReportedAdmission.fp32_add_commute` record remains ACCEPT,
PASS for both gates, with 4096 replicates. Its artifact remains
`experiments/floating_point/report/summary.json#fp32/ADD-COMMUTE`; this report
is no longer a premise of the commutation theorem and has not been replayed.
`#print_fp_assumptions add_value_commuted_equiv` distinguishes the strict model
proof from its empty empirical assumption list.

Both complete kernels succeed and agree on every typed memory cell under
that site condition. Internal registers are private: the candidate additionally
binds `y`, so the theorem does not assert identical register states. The
Python assertions still admit NaN/Inf; the finite-operand condition belongs
to the transformation theorem, not to input admission.
The optional `FPProof.reference`, `LocalAdd10` and `ieee_if_local` helpers
describe software-model conformance under an explicit hypothetical IEEE
premise. They are not public transformation goals or experiment TODOs.

Targeted validation builds both public proof entries and their dependencies
with Lean 4.29.0, including standard-base axiom audits. The assumption audit
separates strict addition commutation from the empty empirical list. The
mask-domain boundary checks also pass, including
arbitrary inactive values and the one-element tail. The kernel theorem, kernel equivalence, canonical
source identities, precision-aware lowering and input assertions are retained.
The task ledger explicitly records removal of the two public host-result
theorems from the required scope. Historical host validation and numerical
report replay remain separate.

`python bench/repairs/tritonbench_g/add_value/check_contract.py` passed exact
Python AST/guard/source-import checks, 11 rejection cases and 14 admission
cases for each kernel. It checks the exact specialization of the original
benchmark AST at 10 and the second kernel's local-constant form, then executes
both JIT ASTs over real CPU backing storage, with
CUDA metadata/allocation/launch mocked, and verifies returned identity and
unchanged input storage. The largest admitted extent is metadata-only.

`python bench/repairs/tritonbench_g/add_value/check_cuda_lowering.py
--output-dir /tmp/veritile-add-value-local-constant-lowering` passed offline Triton 3.7.1
compilation of both kernels for SM80/SM90 at the selected FP32 signature without alignment
hints: predicated scalar `.b32` accesses, signed element-to-byte scaling,
and FP32 addition without explicit FTZ. Both runtime signatures contain only
`x_ptr`, `output_ptr`, and `N`; `BLOCK_SIZE` is constexpr. PTX ISA 8.7 section 6.4.1 requires
natural four-byte alignment for these accesses; ordinary allocation supplies
output alignment. This is compiler inspection, not GPU execution or a
compiler-correctness proof. CUDA was unavailable.

`Misc/TensorBridge.lean` contains the optional Python-check correspondence as an ordinary theorem. `Correct.lean` retains the mathematical kernel specification and `FPEquiv.lean` retains the source transformation equivalence; neither imports the bridge.

`Misc/SourceBinding.lean` retains the source-identity and lowering helpers under their existing `FPEquiv` names.
