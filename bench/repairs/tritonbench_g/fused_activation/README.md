# Fused activation: kernel correctness and bias-addition equivalence

The selected kernel updates x in place. For entry-state logical values, its
independent mathematical result is

```text
z[j] = 0.5 * input[j] + x[j] + bias[j % W]
x_after[j] = sigmoid(z[j])                 # wrapper selection
x_after[j] = max(0, z[j])                  # benchmark's direct ReLU branch
```

Both branches are covered. Input, load, computation, and output use FP32; the
source did not constrain input dtype, so FP32 is the selected default. Real is
only the mathematical interpretation. The native inner multiply/add retains
its FMA rounding; no proof equates FMA with independently rounded operations.

## Inputs and source defects

`Contract.lean` and the Python wrapper retain the same admission: three strided,
contiguous FP32 CUDA Tensors on one device, with no unresolved negative view
bits. Let `N = x.numel()` and `W = bias.numel()`. Input covers the N-element
prefix, and `N,W < 2**31`. For nonempty x, W is positive, every data pointer is
4-byte aligned, and x's live byte interval is disjoint from the full bias and
read input intervals. Bias and input may overlap each other. Shapes need not
match; indexing is flattened with cyclic bias. Contiguous offset views and
singleton dimensions remain admitted. There is no input copying, conversion,
finite-value assertion, or added dtype variant.

Empty x needs no pointer query or launch. The wrapper selects
`min(2048, next_power_of_2(N))` for nonempty launches. It returns the original x;
wrapper execution and returned-object identity are outside required proof scope.
`TensorBridge.python_checks_iff_preconditions` proves formal admission equivalence.
`Proofs/InputBinding.lean` connects all three checked views with the shared
Torch logical reader, including storage offsets and read-only input aliases.
`Proofs/Contract.lean` and `Proofs/Pointers.lean` retain finite index/mask,
cyclic remainder, pointer alignment, storage span, and read/write separation
proofs. These are structural obligations, not numerical deferrals.
The shared [load/store audit](../add_example/LOAD_STORE_AUDIT.md) cites Triton
mask behavior and the PTX scalar-access natural-alignment requirement.

The original benchmark is unchanged. Its wrapper selects block 0 for N=0 and
block 3 for N=3, violating division and Triton arange requirements. Existing
checks also reproduce short input, empty bias, strided-hole corruption,
negative-view mismatch, shifted input alias, cyclic bias alias, and the
original signed-index overflow. These original defects remain separate from
the checked repair; no correctness claim silently limits N to a convenient
multiple or makes a copy to normalize inputs.

## Public kernel proofs

- `Kernels.lean` contains both unchanged Python function bodies directly in
  `triton { ... }`, including the original `activation == "sigmoid"` and
  `elif activation == "relu"` tests. Dtypes and constexpr call metadata are
  outside the bodies. The existing Boolean caller API supplies one of those
  two valid String values; no Boolean test replaces the written Python.
  Generic frontend specialization uses only literal alternatives of the actual
  immutable caller expression and preceding tests on that same name.
- `Proofs/SourcePreservation.lean` proves complete original/candidate kernel
  identities with the prior source on both String alternatives, for every
  pointer, size, multiplier and backend. All prior IO, numerical assumptions,
  failures, successful states and traces remain in scope. The old bodies are
  proof-local compatibility models, rather than the actual source entry.
- The existing
  candidate, which changes only `(multiplier * input + old_x) + bias` to
  `bias + (multiplier * input + old_x)`. The wrapper's multiplier is 0.5.
  Correctness and equivalence share these exact source definitions.
- `Correct.fused_activation_kernel_correctness` is the primary regional
  `GroupedMasked2DKernelIO.RegionalImplements` endpoint: three active
  entry-state reads, cyclic bias addressing, actual masked source access safety, successful
  execution of the canonical source, every active in-place mathematical
  result, and the exterior regional frame. It covers sigmoid and ReLU without
  requiring flat-allocation disjointness. Existing original mathematical
  proofs are reused through a checked projection.
- `TensorBridge.fused_activation_checked_views_correctness` composes that endpoint
  with checked Tensor views read through one shared physical reader. Bias and
  input may be identical or partially overlapping offset views. The conclusion
  includes source execution, logical Tensor output values, and the exterior
  frame. `decode` supplies the mathematical interpretation of stored words;
  this Real theorem makes no exact-native-arithmetic claim.
- `AccessSafety.fused_activation_kernel_access_safe` in `Proofs/AccessSafety.lean`
  retains the explicit masked-access corollary. The public regional IO judgment
  already includes the actual source access-safety proof. It establishes access
  bounds without read-only separation. The optional
  `Flat.fused_activation_kernel_flat_correctness` in `Misc/Flat.lean` retains the prior grouped
  IO contract and its stronger flat-allocation disjointness requirement.
- `FPEquiv.fused_activation_commuted_equiv` proves successful execution and
  complete memory equality for the two typed FP32 sources with `≡[[]]`.
  It uses shared strict addition commutation. The rewrite-site
  domain checks only the two final-add operands on active lanes after the
  actual common load prefix. Arbitrary inactive values cannot affect stores.
- `FPEquiv.fused_activation_native_equiv` in [Misc/NativeEquivalence.lean](Misc/NativeEquivalence.lean) uses the shared
  `ProgramIO.EquivalentOn` judgment to retain the native FMA graph, both
  source-access certificates, successful executions, typed native output equality,
  and both exact active-store frames. `IO.lean` makes the three channels, cyclic
  bias window, in-place output, source invocations, and word interpretation
  explicit. Its logical input-word bindings prevent expression contraction
  across a load. `NativeRewrite.AddBinding ops cfg` identifies only `ops.add32`
  with `FP.add cfg` on FP32 words. The shared strict commutation theorem then
  applies to the **actual FMA result and loaded bias**. FMA and sigmoid ports
  retain their existing interpretations.

The primary regional theorem and its checked-view corollary cover the admitted
read-only aliases directly. They do not rely on the optional flat-memory
contract. `Proofs/InputBinding.bindReadViews` constructs only entry observations
and preserves the program id; it makes no assumption about the kernel result.
Python's x/read separation is the condition needed for the in-place operation
across programs. Full wrapper or allocator verification is not introduced as
an acceptance requirement.

`Proofs/Source.lean`, `Proofs/Grid.lean`, and `Proofs/Graph.lean` reuse the prior
complete source, scatter, and grid results. Historical host, byte-observation,
and software primitive-conformance helpers remain optional. Neither public
entry imports the local Host module. Helpers do not import public proof roots.

## Strict numerical proof and compiler boundary

The only transformation is strict FP32 addition commutation. The source
theorem uses `ScalarArithmetic.add_comm_strict` in the fixed
`Guarded.fp32Theory`, certified by `ScalarOpsLaws.add_comm` for
`Software.strictFP32` with `.ieeeValue`. Its empirical list is empty.
The native theorem uses the same shared scalar proof with its explicit
`AddBinding ops cfg`; it has no `R` or `Models` premise. The chosen `cfg`
retains its NaN, signed-zero and flush behavior. Both theorems preserve the
original finite-operand condition at the actual rewrite site, separate from
Python input admission.
Finite x/input leaves do not establish a finite FMA result. No finiteness of
inactive lanes or activation outputs is required.

The historical report remains intact and is no longer a theorem premise:
`experiments/floating_point/report/summary.json#fp32/ADD-COMMUTE`,
SHA-256 `c00ff0be806504ef7aadd1d85accf7c34303b538a953ffd0cf433dbea1c0e0d4`,
ACCEPT with both gates passing and 4096 replicates. Its scope is unchanged,
and an independent replay has not been run.

The retained `Words.LocalExact` and `Words.LocalExpLowering` helpers describe
optional software primitive interpretations. They are **not** task numerical
TODOs and are not premises of the public rewrite theorems. FP32 rounding,
FMA interpretation, FTZ, literal encodings, exp2 and division ports remain
model/compiler boundaries shared by both versions.

Offline Triton 3.7.1 lowering checks both Python kernels on SM80/SM90, sigmoid
and ReLU, and blocks 4/2048. The observed native graph keeps FMA(input, encoded
half, old_x), bias addition, and either native max or the existing sigmoid
instruction expansion: subtraction from zero, encoded-log2e multiplication,
approximate exp2, add one, and full division. No explicit FTZ is emitted.
The compiler canonicalizes the final bias-add operand order to the same order
in both PTX outputs. The source graph theorem conservatively covers the written
commutation through the strict model binding; it does not claim distinct machine instructions
or a performance improvement. This finite set of compiler observations is not
a proof of the compiler and is not GPU execution.

## Focused checks

```sh
python bench/repairs/tritonbench_g/fused_activation/check_contract.py
python bench/repairs/tritonbench_g/fused_activation/check_cuda_lowering.py \
  --output-dir /tmp/fused-activation-kernel-lowering
```

The CPU check binds both typed DSL bodies to Python, verifies the candidate's
single expression change, preserves the original kernel AST, and passes 16
admission/branch cases, 27 rejections, and eight original counterexamples.
CUDA metadata and launch are mocked. The separate offline compiler check
passes eight configurations; its report explicitly records no GPU execution.

Targeted Lean checks use the existing scoped builder and Lean 4.29.0. Public
proofs pass standard-base axiom audits; `#print_fp_assumptions` separates
strict source addition commutation from its empty empirical assumption list.
Independent mathematical formulas pass noncircularity checks.

`Misc/TensorBridge.lean` holds the supplementary checked-view application, logical-read/pointer aliases and Python-check correspondence. `Correct.lean` has one primary regional IO specification; the optional flat corollary is in `Misc/Flat.lean`, sharing `RealIO.lean` and `Proofs/RealInterface.lean` without importing the reader; scalar derivations and explicit operational access results live in `Proofs/Mathematics.lean` and `Proofs/AccessSafety.lean`. `FPEquiv.lean` retains source equivalence; [Misc/NativeEquivalence.lean](Misc/NativeEquivalence.lean) retains the auxiliary native pair equivalence and its original domain, using strict commutation with the model interpretation described above. `NativeInterface.original_source` is the optional exact-source corollary, outside the main reader. `Proofs/NativeInterface.lean` composes the native full-memory proof with logical input bindings, actual source safety, and the shared IO relation. It retains the raw native result as `NativeInterface.execution_equiv`; neither supporting proof imports a reader entry.


The correspondence manifest and mutation checks pin both complete functions,
including String conditions, their missing final else, slicing, load options
and operation order. Frontend regression checks reject mutable String
constexprs and opaque String values without finite literal alternatives.
No new numerical identity or GPU experiment is introduced by this migration.
