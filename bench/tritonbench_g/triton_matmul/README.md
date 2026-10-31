# triton_matmul

The current implementation is [the repaired Python kernel](triton_matmul.py).
[Sources/Kernels.lean](Sources/Kernels.lean) contains its unchanged function body
inside `triton { ... }`. Pointer dtypes, nine runtime integer arguments and four
constexpr arguments are declared outside the pasted body.

[Real correctness](Correct.lean) proves the full matrix product over every K
coordinate, with access safety and an exact exterior frame for every complete
grid permutation. Both FP16 and E4M3FN storage, FP32 accumulation, signed grouped
indices, clamped tails, the runtime loop bound and both output conversions are
retained. Launch facts used by the proof follow from the existing host configuration.

[FP equivalence](FPEquiv.lean) compares this source with the existing
[group-eight shift candidate](CandidateKernels.lean). The candidate changes only
`GROUP_SIZE_M * num_pid_n` to `num_pid_n << 3` when the group size is eight;
other group sizes use the same repaired body. No floating rewrite assumption is used.

[Proofs](Proofs/) decompose and execute the actual source. The independent
mathematical recurrence and finite-index refinement support these proofs;
previous complete Lean kernel quotations and their old execution proofs are removed.
The explicit output cast and implicit store conversion are both present in the IR.

The [correspondence manifest](jit-correspondence.json) checks all 29 top-level
Python statements, including the configured JIT decorator and dtype branch.
[Current validation records](../../../validation/) distinguish source checks,
Lean proofs, pending corpus translations and historical numerical receipts.
