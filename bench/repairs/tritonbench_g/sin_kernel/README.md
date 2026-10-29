# sin_kernel: checked FP32 kernel and index variant

The independent formula is `out[i] = Real.sin(x[i])`. The selected production
profile uses FP32 loads, `tl.math.sin`, and stores. `Kernels.lean` contains the
complete readable typed DSL for the original and the retained candidate. The
candidate changes only `pid * BLOCK_SIZE` to `BLOCK_SIZE * pid`; it changes no
floating expression. Both public roots use these exact source definitions.

The required proof scope is the kernel. Python supplies input assertions;
allocation, launch orchestration, device selection, and returned-object identity
are outside the public acceptance claim. Earlier whole-wrapper results remain
optional in `Misc/Legacy/HostCorrect.lean`, `Misc/Legacy/HostFPEquiv.lean` and their retained
helpers. The original benchmark Python is unchanged.

## Contract and memory meaning

The wrapper accepts a strided FP32 CUDA Tensor that is contiguous in the default
format, has no unresolved negative view bit, and contains fewer than `2**31`
elements. Nonempty data pointers must be divisible by four. Empty inputs do not
query the pointer. Scalar tensors, offset views, unusual singleton strides,
NaN/Inf, signed zeros, and subnormals are admitted. No value-finiteness condition
is imposed on input admission or on the index transformation.

The eight Python checks match `Contract.lean` exactly, including non-Tensor
rejection. `TensorBridge.python_checks_iff_preconditions` proves admission in both
directions. The unchanged repaired wrapper allocates `empty_like`, returns it
immediately when empty, and otherwise uses BLOCK1024 on the input device. It
performs no input copy, layout conversion, or dtype expansion.

`Correct.sinIO` declares input/output windows `pid*1024 + lane` and their shared
`offset < n` mask. Its flat-memory meaning quantifies over disjoint placements
whose active lanes fit the allocations. It preserves every cell outside the
active output window; inactive tails need no padding allocation. This statement
starts from memory/program-ID conditions and does not assert a Torch wrapper
execution. Shared Tensor laws and the retained optional bindings explain how
contiguous logical elements correspond to raw FP32 storage and storage offsets.

## Counterexamples and structural bounds

The original direct pointer source does not implement arbitrary strided or
unresolved-negative logical views. CPU backing-storage replay reproduces both
mismatches. At `n = 2**31`, its signed-int32 extent becomes negative and all masks
are false despite a nonempty grid; the original `all_masks_false_at_two31`
theorem and counterexample remain available. The explicit contract rejects these
cases. The admitted extent and grid prove every launched offset, including tail
lanes, stays below `2**31` and gives the intended signed mask.

Contiguity does not guarantee alignment: actual CPU FP32 views over byte-offset
storage yield pointer remainders 1, 2, and 3. The guard rejects those nonempty
views, while aligned offset views and empty views remain admitted. Their CUDA
construction has not been tested.

## Public proofs and numerical scope

- `Correct.sin_kernel_correctness` states the independent `Real.sin` formula,
  successful finite flat execution, active access bounds, and the complete
  exterior frame. `Proofs/Projection.lean` connects the canonical source's
  mathematical interpretation to the existing valid kernel proofs.
- `FPEquiv.index_commuted_equiv` proves both displayed FP32 kernels succeed and
  agree on every typed memory cell from arbitrary initial states. The proof
  actually commutes the integer prefix and retains the full load/sine/store
  continuation. It is not identity substitution for a missing candidate.
- `FPEquiv.original_source` and `candidate_source` bind the public programs to
  the canonical definitions. The FP interpreter retains explicit FP32 loads
  and operation tags. Mathematical projection is used only in the Real proof.
- The domain and numerical assumption lists are empty. Arbitrary masked-off
  load values, including nonfinite values, need no restriction; active FP32
  values likewise need no numerical condition for this index-only change.

`#print_fp_assumptions index_commuted_equiv` reports **none**. There is no pending
numerical transformation experiment. Optional `LocalSine` and FP reference
transport helpers remain hypothetical primitive-conformance results, not public
acceptance goals or experiment obligations. No correctly rounded sine, native
accuracy, native-to-IEEE conformance, or FP-to-Real equality is claimed.

## Validation

Targeted Lean builds use `/tmp/veritile-lean-toolchain` with
`LEAN_NUM_THREADS=2`; public theorems pass standard-base axiom audits.

`python bench/repairs/tritonbench_g/sin_kernel/check_contract.py` checks exact
original AST preservation, the candidate's sole integer-product swap, the Python
assertions, canonical source imports, and both JIT bodies over CPU backing
storage. Each variant passes 11 rejection and 12 admission cases, including
scalar/empty/tail/multiblock shapes, offsets, nonfinite values, all three byte
misalignments, unchanged input storage, and arbitrary inactive NaN loads. It
retains the original layout and signed-boundary counterexamples.

CUDA allocation and launch are mocked. No GPU execution or new compiler
experiment was performed. The CPU sine oracle checks indexing and storage;
it is not evidence for GPU sine accuracy.

`Misc/SourceBinding.lean` retains the source-identity helpers under their existing `FPEquiv` names.

Optional proof layout (2026-10-21): `Misc/SourceBinding.lean`, `Misc/TensorBridge.lean` are optional entries in `Misc/`. The required `Contract.lean`, `Correct.lean`, `FPEquiv.lean` and their proof helpers do not import these entries. Declaration names, statements, proof terms and acceptance status are retained; this move adds no mathematical completion or numerical experiment.
Historical wrapper readers are physically retained in `Misc/Legacy/HostCorrect.lean`, `Misc/Legacy/HostFPEquiv.lean`. They remain optional historical interfaces and are checked separately from the current primary/auxiliary import union.
