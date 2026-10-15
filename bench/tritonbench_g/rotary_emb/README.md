# rotary_emb

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

- Source file: `rotary_emb.py`
- Corpus: TritonBench-G v1
- Status: full source, real/FP proofs and typed numerical regression compile and pass explicit axiom audits; independent trust replay remains pending.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

`Kernels.lean` retains the full 34-statement source. `Correct.lean` states
independent real contracts for both original and candidate, with successful
execution, all Q/K even/odd results in the same final state, source-mask access
safety, and raw-cell frames in region and flat memory. `FPEquiv.lean` proves the
typed implementation equivalence without numerical atom assumptions.

The real formulas are explicit in `Correct.lean`: even outputs use
`x_even * cos_even - x_odd * sin_even`; odd outputs use
`x_even * sin_odd + x_odd * cos_odd`, all from initial input memory.
`Proofs/Witness.lean` supplies a disjoint flat allocation and a nonempty,
collision-free two-sequence/two-head/two-pair strided execution witness for every
storage choice and both variants. Region contracts allow arbitrary undefined
values; only the current flat-memory bridge requires zero undefined values.

The historical `rotary_emb_kernel_correctness` declaration is retained verbatim
in `Correct.lean`, with supporting proofs in `Proofs/LegacyReal.lean`. It concerns four separate store slices with their own
sine strides and even-index trigonometric windows, including for the odd store.
It does not establish source correspondence for the full kernel. The full
contracts use the source details below. All 60 historical named declarations,
their original documentation and the supporting source are preserved; only
the headline's physical module location changes.

The final dependency-aware build of `Correct.lean` and `FPEquiv.lean` passes
(3512 jobs), reaching all 73 local Lean modules. `Correct.lean` imports the
complete typed numerical execution, so it is checked with the public entries.
Its closure includes all 34 statements, both Q/K output pairs,
the raw memory frame, actual source trace safety, all-profile typed transport,
flat-memory contracts and both finite execution witnesses. Explicit checks of
twelve public, historical, transport, witness and numerical declarations find only the
standard Lean axioms `propext`, `Classical.choice` and `Quot.sound`. FP
assumptions are empty, and both public formulas pass non-circularity checks.
These compiler and statement audits do not replace independent comparator replay.
All three public coverage declarations are migrated with validated source/support
fingerprints. Both generated specification sheets have no self-reference or
missing summary; final corpus acceptance remains pending.

All 216 three-storage templates now compile. The production DSL resolves
synthetic metavariables and instantiates their assignments before returning the
expanded term, avoiding the previous declaration-processing memory growth.
The complete shared-library refresh passes 3525 jobs, and the full symbolic
34-statement regression `bench/tests/RotaryEmbElaboration.lean` passes in 5.08
seconds with 3,826,196 KiB measured child peak RSS. Precision checks retain both
raw fp32 carriers and real carriers with explicit fp32 compute tags while
distinguishing fp64; no storage profiles were removed.

The separate exact-real-backend numerical regression passes for both mixed
storage profiles and both implementations. `Proofs/ExecutionQ.lean` and
`Proofs/ExecutionK.lean` prove successful execution of the actual consecutive
source blocks; `Proofs/Execution.lean` composes them and uses the checked
candidate equivalence. Q's checked even/odd results are -7 and 25; K's are -13
and 79, including the second K head. Head-masked, sequence-masked and stride-gap
cells retain 7, and the sine cell is unchanged. Nonzero program IDs, independent
nonunit strides, differing odd/even trigonometric values and both store masks
are retained. The Q and K modules compile in 232 and 41 seconds; the final
composition and readbacks compile in 48 seconds. Register and memory frame
lemmas avoid repeated expansion of the Q state. The model is not IEEE rounding
certification.

The full source has several details that the slice theorem does not establish:

- Sine loads use the **cosine** strides. The two sine-stride arguments are unused
  in the Python kernel and remain unused in the transcription.
- Even and odd results read their corresponding even and odd cosine/sine values.
- Both Q stores precede K loads and the second set of trigonometric loads.
- Only `2 * floor(BLOCK_DMODEL / 2)` logical dimension lanes are generated.
  No lane is generated for an odd last dimension. Its physical cell is preserved
  when it lies outside the actual address footprint; arbitrary overlapping strides
  can alias a different generated lane to that cell.
- Trigonometric loads have shape `[BLOCK_SEQ, 1, BLOCK_DMODEL / 2]` and a sequence
  mask only. They can execute even when every data head is masked.

The new model factors 216 three-storage templates into independent Q/K/Cos/Sin
storage choices. The unchanged templates are defined in 36 six-profile helper
modules under `Proofs/SourceProfiles_*.lean`, sharing `Proofs/SourceTypes.lean`,
so each successful profile group produces a reusable artifact. This is structural
interpreter coverage, not native compilation certification for every combination.
Offline lowering with Triton 3.7.1 for CUDA sm90, `BLOCK_HEAD=4`, `BLOCK_SEQ=16`,
`BLOCK_DMODEL=64`, and four warps succeeds for homogeneous fp32/fp64/fp16/bf16.
The unchanged source **fails native lowering for homogeneous fp8e4nv and fp8e5**:
`llvm.fmul` receives an `i8` operand in `ConvertTritonGPUToLLVM`. The scoped receipt
is `Proofs/native_lowering_sm90_triton_3_7_1.json`; reproduce it with
`TRITON_CACHE_DIR=/tmp/veritile-rotary-native-cache python3 bench/tests/RotaryEmbNativeLowering.py --output /tmp/rotary-native.json`.
The probe exits 1 when a profile fails. It performs no GPU execution or numerical
comparison, and does not certify mixed-storage profiles or other configurations.
The candidate only commutes an exact natural-number head-offset
multiplication. Numerical arithmetic and all four stores are retained.
The promotion transcription follows the pinned
[Triton 3.5.1 semantic implementation](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/semantic.py#L60-L98):
each multiplication promotes its own operands before the final addition or
subtraction promotes those two results. Storage conversion is explicit when the
computed result dtype differs from Q or K storage.

The real formula uses initial Q/K and trigonometric memory. Its domain
requires distinct Q/K/trigonometric output regions (Cos and Sin may alias),
collision-free active even/odd output addresses, and source-mask-specific access
bounds. Strides remain independent nonnegative naturals; negative strides,
machine integer overflow, native `arange` shape restrictions, and valid launches
are separate backend boundaries. The host head grid is based on `HEAD_Q`; a
per-program theorem alone does not certify coverage of extra K heads beyond that
grid.
