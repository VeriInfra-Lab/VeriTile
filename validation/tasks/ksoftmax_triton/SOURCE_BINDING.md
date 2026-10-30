# Binding to the selected Python

Candidate source: `ksoftmax_triton_grid_y_checked.py`.
Frozen predecessor: `../ksoftmax_triton/ksoftmax_triton_repaired.py`.

The sole delta is the two-line `x.shape[1] > 65535` rejection appended to `_extent`.
Both public wrappers reach `_extent` only after all original header and mask
checks and the original empty return. The added guard runs after the original
integer extent predicates and before any live data-pointer query. Actual launch
grids remain `(B,Q)`. The existing `B < 2**31` extent predicate supplies the x-axis
limit; the new guard supplies the y-axis limit, and live rank-three numel supplies
axis positivity. No launch-product predicate is assumed or checked.

The `checkForward` and `checkBackward` definitions in `Contract.lean` copy the original metadata
check order with exactly that new branch. Their exact-bool `flags` input records
classification of `log` and `causal`; `ArgumentMetadata` distinguishes tensors and
non-tensors, including ignored `M`. Exact-domain theorems preserve all old layout,
contiguity, dtype, common-device, rank/shape, mask, alignment and separation rules.
Empty calls bypass both old extents/physical rules and the new cap, as in Python.

The `forward` and `backward` declarations in `Kernels.lean` alias the original complete typed
quotations, with definitional equality witnesses. The common whole-module AST and
byte delta check binds both unchanged JIT definitions, widened index casts,
heuristics/autotune declarations and dispatch options to that frozen source.

All new public mathematical/native adapters call the corresponding frozen public
kernel theorem with `admitted.1` and pair its entire result with the derived
`SupportedLaunch`. Their result includes complete source execution success,
every output, and every original typed memory/physical byte frame. The candidate
introduces no assumed body-success proposition, load-binding premise, numerical
rewrite port, allocator proof, or host-framework execution gate. The unchanged
selected local compiler and ordinary Tensor model hypotheses remain explicit.

The five new modules and audit import the frozen original modules; no old proof
is edited or rebuilt. Proof, source, receipt, effective import resolution and
historical evidence pins are listed in the separate delivery packet.
