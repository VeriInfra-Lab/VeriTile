# Kernel and Tensor interfaces

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `Misc/NativeCorrect.lean`, `Misc/Auxiliary.lean`.

Current source-pair review (2026-10-21): The current `FPEquiv.lean` exposes actual forward/backward source versus the already specified `address_commuted` variants of the same finite-seed repair macros. Both inner/noninner choices and every one/multiple-tile branch are retained, with exact FP32 operations and casts. `PairIO.lean` makes the source programs explicit; `Proofs/SourcePair.lean` reuses the original exact integer structural lemmas. No numerical rule or extra operand domain is used. The previous full addressed-native `kernel_forward`/`kernel_backward` declarations keep their full names, statements and proofs as ordinary theorems in `Misc/NativeCorrect.lean`. All native aliases and consumer reductions remain auxiliary. No new full-grid pair or hardware claim is made.

`Correct.lean` presents the forward softmax and backward saved-output VJP through
shared finite `ProgramIO` judgments. `IO.lean` exposes the complete tensor ports,
logical offsets, masks, precision interpretation and actual source grid. Forward
permits finite values and negative infinity, with a finite value in each normalized
column; the original output-separation and launch-geometry conditions are explicit.

`Misc/NativeCorrect.lean` uses shared `AddressedProgramIO` for the independent native graph.
`NativeIO.lean` exposes fixed stored FP32 words and each output consumer's reduction
and lowering schedules. All FP32 words and source/output aliases remain admitted.
The initial output words are an explicit finite read port; addressed binding
enforces alias coherence without an added separation premise. Primitive operations
and schedules are model parameters. No native=Real claim, numerical transformation
assumption or numerical experiment is introduced.

Both interfaces include safety of the actual evaluated source accesses, successful
execution of every original grid program, every output and the exact exterior
memory frame. All eight branches (forward/backward, inner/noninner, one-tile/streaming)
are covered. `Proofs/` connects the existing execution proofs to these interfaces;
`Misc/Auxiliary.lean` retains the exact prior raw statements as ordinary theorems,
including full term-memory equality and observations outside the output tensor.

`../Misc/TensorBridge.lean` retains all checked Tensor applications,
actual input-byte loads, output words, caller bytes, full input-backing frames and
empty cases. Its only change is an explicit import of the existing frozen v1
Correct library used by its two Real applications; all statements and proof bodies
are unchanged. The original v1/v2 sources, Python guards, contracts, public roots
and evidence snapshots remain unchanged.
