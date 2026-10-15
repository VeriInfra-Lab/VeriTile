# Source and execution scope

The original Python file has SHA-256
`e145461bdb74f2d5ae5de59275181e6e22d39f0b0b133cd191d214ec5ab4f40a`.
The selected JIT kernel is `chunk_global_reversed_cumsum_scalar_kernel`
from `thunlp/TritonBench@603e28a`. `Proofs/Source.lean` retains the complete
original loop, boundary load without a padding option, reduction, carry
update, forward inclusive scan, output conversion and boundary store.
The candidate changes only `i_bh * T` to `T * i_bh` in the input pointer.

Reverse iteration uses a signed mirror index. Its signed product with BT is
converted to a natural offset after multiplication; T = 0 executes no loop
iterations. Input and output storage dtypes remain independent. The output
pointer's `dtype.element_ty` determines the explicit final conversion.

The real result at time t in row r is the sum of the initial input values
from t through T - 1, plus the sum of the initial unspecified-value oracle
from T through `ceil(T / BT) * BT - 1`, using address `r * T + k`.
Every valid output contains that same tail contribution. The ordinary suffix
formula holds exactly when the total tail is zero; aligned lengths suffice.
In particular, the original Python example T = 4 and BT = 16 does not itself
justify a zero-padding assumption.

The complete per-program real execution requires distinct input/output
regions and positive BT. Registers may initially contain arbitrary values.
The specification includes successful execution, full tagged output cells,
and preservation of every tagged cell outside the active output row.
Finite IO additionally requires bounded active addresses, a disjoint finite
allocation covering all positive extents, and an explicitly zero unspecified
oracle, as required by the current generic flattening theorem. The region
theorem retains arbitrary oracle values.

Typed whole-state equivalence quantifies over every arithmetic backend,
initial state, T/BT and independent input/output dtype, including failure.
No floating algebraic law or numerical domain guard is used. A concrete
mathematical consistency backend supplies real tile reductions and scans;
its identity casts do not model IEEE rounding. Its complete typed success
witness starts with empty registers, bf16 inputs, fp16 output, row 1,
T = 17, BT = 16 and unspecified padding value 7. The public FP interface
uses the same `FP.ProfiledRewrite.engine` as this witness.

The full source AST check and seven scalar mutation rejection controls are
recorded in `bench/review-checkpoints/2026-10-13/reverse-source-ast`.
Offline native admission is recorded in the sibling `reverse-native`
checkpoint: 150 scalar cases cover all five original BT/warp configurations
at six matching storage types for T = 0, 1, BT and BT + 1, plus all independent
storage pairs at BT = 16, two warps and T = 17. These checks compile unchanged
original JIT bodies, decorators and wrappers with Triton 3.8.0 for sm90.
They establish admission only for the recorded cases; they do not execute
GPU kernels or prove numerical refinement.

The scalar JIT uses linear row addresses and reads no tensor strides.
The current theorem therefore does not establish the host wrapper for
arbitrary noncontiguous inputs or `empty_like` output layouts. A host layout
ABI, launch-wide composition, autotuner behavior, native fixed-width index
arithmetic and its no-overflow domain, and IEEE/native numerical refinement
remain outside the proved per-program contract. Independent isolated source
comparator replay also remains pending. Source acceptance is incomplete.
