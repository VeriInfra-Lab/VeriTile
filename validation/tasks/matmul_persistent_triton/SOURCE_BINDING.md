# Source binding

Candidate Python SHA-256:
`cbc4b8cf640d36422da090fe981e5d84a1d5a7f3e686e6b7842a43271ef875b6`.
Frozen repaired Python SHA-256:
`ad6eb70f7493aff7f323558870765887e09c208c3a547e2c6271fe519efddd7d`.

The changed scalars are M, N, K and strides am, ak, bk, bn, cm, cn. The exact
full-module reversal and the actual binder report bind this sibling to the
frozen source. No unit-value exclusion, disabled specialization, altered
configuration, or smaller metadata domain is introduced.

The readable quote preserves the source's two distinct dynamic loops,
grouped tile coordinates, clamping before pointer formation, signed K-tail
mask, supplied accumulator dot, output conversion and masked write. Exact
boolean increments are rendered as conditionals. Host scalar widening is
represented by Nat parameters and explicit Int conversion at address sites.
`Pointer.scalar_bounds` derives representability of all nine scalar casts
from ordinary valid Tensor ABI bounds; `Pointer.*_offset_pointer` proves that
every formed clamped DSL offset uses the actual Tensor data pointer without
unsigned64 wrap. Frozen `loop_bounds`, `group_intermediates`, `k_control` and
`formed_addresses` supply the corresponding exact control/address bounds.

This is an explicit reviewed Python-to-readable-DSL transcription plus
source/binder evidence, not a proved Python parser/compiler. Once inside the
quote, the execution chain is proved using the shared structural interpreter.
No equality from the entire quote to `Source.run`, `Source.step`, or another
handwritten event program is assumed. The old `Source.accumulator` is used
only as a proved mathematical intermediate toward the independent target.

`Core.backend` is defined on shared primitive operations. Unused primitive
fields have concrete defaults. The quoted program uses only its literal
zero, scalar storage conversion, and accumulator dot fields. The Real dot is
the finite-sum operation. The native dot explicitly expands into four
ordered fixed-size ISA stages with the actual incoming FP32 accumulator.

The complete grid theorem derives register readiness from the actual quote
prefix. It composes exactly `range(grid)` and each actual persistent ordinal
range. Readiness, coverage, and memory preservation are derived internally;
they do not restrict public shapes, strides, unit values, or stored words.

`Public.invoke` starts after ordinary allocation, with the actual allocated
Tensor supplied through `TensorBinding.Arguments`. It does not assume host
or allocator numerical correctness. Metadata/allocation observations remain
ordinary representation contracts. The existing guards, allocation/device
context and call strategy are pinned by unchanged Python and CPU AST checks.

The 2026-10-21 source pair adds `matmul_persistent_triton_reordered.py` with
only the common signed row-index multiplication commuted. Both alternatives
have complete ordinary quotations for FP16 and E4M3FN in `Kernels.lean`.
`Pair.selected_original_preserved` checks the original macro expansion by
definitional equality, while `Pair.kernel_equal` and `Pair.equivalent` prove
actual full-source and complete persistent-grid equality, respectively.
The public main theorem uses `PairIO.equivalent` and the shared complete
`ProgramIO` judgment: original input binding/bounds, both actual safe and
successful schedules, equal output cells and both exact exterior frames.
The complete source-delta check pins all other Python AST and quote content.
`Misc/CandidateCorrect.invoke_equal` transports this pair through the retained
Tensor wrapper and physical output materialization, including both no-launch
branches. This introduces no numerical experiment or new Tensor/API premise.
