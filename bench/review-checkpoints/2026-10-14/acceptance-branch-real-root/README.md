# Recurrent branch integration — passed

This immutable snapshot builds the complete public Correct entrypoints for chunk_gated_attention and chunk_gla_fwd, plus chunk_gla_fwd.FPEquiv. All three targets and their dependencies completed with exit code 0; 273 captured inputs remained unchanged. See receipt.json and build.log.

The shared gated recurrence now follows the actual conditional AST and proves that moving its common continuation preserves execution and memory access safety. The two chunk GLA FP guard annotations match their actual fp32 compute context; FPWhereContext proves equality of the affected operand guards.

This is a selected Lean build result, not full-corpus acceptance or independent comparator replay. Other tasks and the host sandbox remain blocked as described in the current review bundle.
