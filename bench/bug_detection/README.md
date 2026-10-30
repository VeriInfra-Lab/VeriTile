# Bug-detection equivalence problems

65 independent Lean problems: 30 negative controls, 20 algebraic rewrites and
15 parameter variants. [Browse the problems](TASKS.md).

The [collection audit](../IMPORTED_KERNEL_AUDIT.md) records confirmed conversion
defects and remaining proof obligations separately from compilation.

Each problem keeps its own complete source and interface:

- `Kernels.lean`: readable `triton { ... }` original/candidate lowerings with
  explicit storage types, casts, masks, arithmetic order and full output scope.
- `IO.lean`: the complete memory layout, scratch footprint, launch, named
  mathematical arrays and input/output coordinate equivalences. The FP view
  derives its addresses and sizes from that layout and retains storage types.
- `Contract.lean`: independent Real formulas and the allowed input domain.
- `Correct.lean`: the `Spec.Real (IO ⊨ fun ... => ...)` problem, including full
  execution, safety, output and exterior-frame obligations. Label-dependent
  losses use the restricted-input version of the same judgment.
- `FPEquiv.lean`: `original M Allowed D ≡[R] candidate N Allowed D`, its
  non-derivability problem, and a separate exact typed-output counterexample.
- `Proofs/Backend.lean`: unsolved connections to supplied native source schedules.
- `Misc/Validation.lean`: interface and coordinate checks.

There is no `Shared` directory, custom tensor interpreter, or cross-problem
import. Each problem uses only its own files and the project's general Triton,
memory, FP and Mathlib APIs. The optional aggregate modules import all problems.

These are **unsolved problems**. The explicit Triton programs are lowerings;
their correspondence to native fused implementations is **pending**. Equal
lowered arithmetic with distinct native backends still requires those backend
connections. Primitive models `M` and `N`, input restrictions `Allowed`, operand
domain `D` and numerical rule set `R` remain explicit. No native whole-kernel
execution callback or new numerical admission is introduced. Compilation checks
the declarations and projection/coordinate witnesses; it establishes neither
the Real/FP goals nor native correspondence or numerical acceptance.

The complete shape is fixed per problem in `task.json`. Probe scales are metadata,
not extra input restrictions. Finite attention sentinels remain finite in the
Real formula; they are not replaced by zero masked probability. MoE keeps top-k
selection, tie policy, selected-token GEMM sizes, capacity order and probability
placement. A native tie policy must be connected explicitly or its input domain
restricted. Native primitive approximations and backend lowering remain proof
obligations.

With the pinned toolchain and project dependencies installed:

```bash
lake build BugDetection
python3 bench/bug_detection/check.py
```

For independent builds, supply a library directory containing the compiled
`VeriTile` and `Mathlib` dependencies (and no bug-detection modules):

```bash
python3 bench/bug_detection/check.py --lean /absolute/path/to/lean \
  --library-path /absolute/path/to/libraries --record
```

Add `--task rope_interleaved` for one problem. Full builds compile every problem
in isolation before checking aggregate imports. `validation.json` records source
hashes and the exact compiler version; `--verify-record` detects stale evidence.
The checker and `interface_review.json` enforce layout and record scope, not
mathematical correctness or semantic fidelity. Proof and native-binding statuses
are tracked independently in the manifest and each problem's metadata.
