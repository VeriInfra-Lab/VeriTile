# Parallel attention complete execution build

This receipt builds `bench.tritonbench_g.parallel_attention.ParallelAttention`
from the final frozen source and current integrated DSL dependencies. The source
contains both complete execution proofs and the audited-name regression
`pa_unspecified_padding_correct`; it retains the original two source surfaces,
all source real-carrier casts, and every `allow_tf32=false` dot option.

The initial state's address-indexed unspecified padding is an independent input
of all eight guarded formulas, including masked `dz`. The forward hypotheses
remain unchanged. The backward theorem removes the old global zero-undefined
premise. No numerical axiom, zero-padding condition or desired-output assumption
was added. Exact before/after statements, references, staged diagnostics and the
independent complete-forward regression are in `../parallel-attention-contracts`.

`inputs.json` and `inputs/` bind every recursive project dependency, the module,
lakefile and toolchain. The receipt's input-change check must be empty. The
separate imported-module audit in `module-audit/` checks the actual compiled
module's entire theorem inventory, its three headline declarations, and the
noncircularity of the four independent output formulas.

This is local Lean elaboration and axiom/dependency checking. It is not GPU
execution or official independent comparator export/replay. The existing
named-correctness-surface/frame and backward helper/descending translation
review boundaries remain; see the port README and proof blocker metadata.
