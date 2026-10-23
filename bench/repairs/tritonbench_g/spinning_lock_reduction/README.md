# spinning_lock_reduction

`Correct.lean` contains the public `EventProgramIO.ImplementsOn` specification; `Misc/NativeCorrect.lean` retains the ordinary native theorem, parameterized by the producer or consumer stage. `IO.lean` exposes the quoted source, its actual decoder, separate partial/output/lock regions, finite inputs and their in-place outputs. `Formula.lean` gives the independent stage formulas and publication/readiness domain. The pure finite producer-to-consumer domain connections retain their names in `Misc/RealStageConnection.lean` and `Misc/FPStageConnection.lean`.

The partial input is the `numSms * bm * bn` prefix, the lock input is the `numSms` prefix, and the output input includes its complete allocated window. Producer-owned partials become zero and their locks become one; consumer stages preserve partials and locks. Each stage writes zero exactly on its independent output rectangle, preserving other output cells. Padding and all exterior memory remain arbitrary and unchanged. `Proofs/RawIO.lean` proves that every input admitted by the previous raw theorems is represented by this finite boundary.

`Proofs/ConcurrentSource.lean`, `ConcurrentControl.lean`, `ConcurrentRun.lean` and `ConcurrentKernel.lean` connect the existing source compiler's extracted plan to the shared closed event/controller interpreter in `VeriTile/Triton/Concurrent/`. They preserve the original PID interleaving, nine rounds, current-memory CAS and its returned old word, explicit barrier, signed/unsigned 64-bit operations, and every actual access. Completion is constructed from the existing source results without a fairness assumption. `Proofs/IOBoundary.lean` supplies a concrete capacity witness; `IOImplements.lean` transports the safe execution from finite canonical inputs to arbitrary exterior memory and proves outputs and frames.

The native result uses the existing FP32 scalar model and integer locks. The needed `+0 + +0 = +0` law is proved in that model. No numerical assumption or experimental obligation was added. Existing compiler/GPU refinement boundaries remain unchanged.

`Proofs/RawCorrect.lean` and `RawNative.lean` preserve the previous raw statements and proof bodies, including the result for every completed producer execution. `Misc/TensorBridge.lean` retains its complete logical/physical-byte statements and proof bodies, importing those supporting modules. Neither main entry imports `TensorBridge`. The original Python, source quotation, guards and `Contract.lean` remain unchanged.

## Tile-area implementation pair (2026-10-21)

`spinning_lock_reduction_area.py` supplies a complete callable two-phase
candidate. `Kernels.areaKernel` is its complete readable source quotation.
Both partial-address expressions change from `pid * BM * BN` (or `next_pid`)
to `pid * (BM * BN)`. BM and BN are constexprs; computing their product first
removes one live integer multiplication. No measured hardware speedup is claimed.

`FPEquiv.tile_area_equiv` compares the actual original and candidate stages
with an empty experimental rule set. `Proofs/AreaEquivalence.lean` proves
64-bit modular address equality, exact controller-step equality and exact
execution equality for every finite thread schedule, including disabled PIDs,
all CAS failures/retries, returned old words, barriers and all nine rounds.
It also preserves completed executions in both directions. The public concrete
source view in `ProgramView.lean` requires successful decoding, completed safe
runs with the same complete state, full observations and exact exterior frames.
Original `IOProof.implements` supplies their existence under the unchanged
finite publication/readiness domain and buffer conditions. There is no new
finite-value premise, CTA serialization or numerical relaxation. Original
independent Real and optional native/Tensor declarations remain unchanged.

The complete Python AST delta check verifies just the two proved address
changes and candidate entry naming; integer wraparound checks include large
64-bit words. These checks are not a floating-point experiment.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `Misc`:

- `Misc/NativeCorrect.lean`
- `Misc/FPStageConnection.lean`
- `Misc/RealStageConnection.lean`
- `Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.
