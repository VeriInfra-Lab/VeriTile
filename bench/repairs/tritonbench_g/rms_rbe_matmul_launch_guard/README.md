# RMS matmul candidate with explicit launch-axis admission

Current FP entry status (2026-10-21): complete actual-program pair. `FPEquiv.source_candidate_equivalence` compares the complete original grid with the original-family output-row index candidate, instantiated on the repaired source. The only source change is final `pid_m * BM` to `BM * pid_m`. `rms_rbe_matmul/Kernels.lean` displays both complete FP16/E5 profiles and both complete candidates directly. `SourcePair.readable_preserves` proves complete equality to the previous shared original, including ports and every typed instruction; the historical `kernel` alias remains unchanged for proof clients.

`PairIO.original` and `PairIO.indexCommuted` retain the original logical channels, signed-byte storage domain, actual program IDs, every program of the rectangular grid and the empty-grid behavior. `PairIO.equivalent` proves both safe successful executions, equality of every active typed output cell, and both full exterior frames. The integer rewrite holds for arbitrary scalar backends and all floating input words, including nonfinite values. The main pair uses the existing native symbolic operations; no numerical assumption or deferred numerical experiment is used. `Misc/CandidateNative.lean` applies the unchanged instruction/profile observer to the candidate's identical symbolic outputs.

This is a separate candidate for the main `rms_matmul_rbe_wrapper` API. The
previous canonical Python file and its frozen kernel proof packet are unchanged.
The old wrapper admits `B=1, M=1048561, K=1, N=2, heads=1, head_dim=2`, with
`use_rbe=False` and `start_pos=0`; its actual grid is `(1, 65536)`.

The candidate adds one guard before the original conversion and allocation:
when both batch and rows are nonzero, batch must be at most `2**31-1` and
`cdiv(M,16) * cdiv(N,64)` must be at most `65535`. These are the two axes used by
the original launch. Empty calls retain the original no-launch behavior. The
guard imposes no product-of-grid-axes bound. Kernel code, launch geometry,
conversions, numerical order, original shape/alias checks, and unused raw helper
code are unchanged; AST equality is checked automatically.

`Contract.lean` proves that the candidate checks are equivalent to the original
input contract plus this explicit nonempty launch condition. The concrete old
admission and new rejection are proved in `Proofs/Counterexample.lean`.
`Proofs/Launch.lean` connects these axes to the unchanged kernel grid and proves
that every program ID survives the native signed-32 interpretation exactly.
`rms_rbe_matmul_grid_real` and
`rms_rbe_matmul_grid_native` expose genuine shared `ProgramIO` records,
independent formulas, the Real specification and an ordinary native-observation
theorem. The records show X, W and RMS channels, dense input/output
addresses, FP16 or signed E5-byte input storage, FP16/FP8 output storage, every
program of the original grid and no private scratch. Their shared judgment
proves actual source access safety, successful execution, all typed output
values and the complete exterior cell frame, including empty launches.
`Misc/TensorBridge.lean` retains both full returned-Tensor theorems, native byte and
input backing frames, and the checked native program-ID correspondence. Its
launch admission is derived from the unchanged candidate input contract.

`check_launch.py` executes the actual wrapper and guard ASTs against Tensor
metadata API doubles and records the requested grid. It saves the concrete old
counterexample, the candidate rejection before allocation, and 22 boundary
cases covering both storage formats, each axis, a large legal grid product,
and empty inputs. These controls do not allocate Tensor payloads, execute JIT
kernels, or run on a GPU. Existing numerical/source/offline-lowering evidence is
reused because every kernel/helper AST is identical. Ordinary successful
Tensor/allocation APIs and foundational scalar/conversion/K8 MMA semantics retain
their previously documented boundaries; the known launch-axis defect is now
checked explicitly for this candidate's main API.

The primary files do not import TensorBridge. Both layers share the same existing
raw grid execution and mathematical/native correspondence proofs. Tensor dtype,
shape, device, pointer identity and preparation remain in the optional bridge.

The old arbitrary-state specifications are ordinary theorems in
`Misc/RawEndpoints.lean`, with their complete original statements and
all-offset `ByteMemory` premise preserved. `PublicRawTransport` extracts their
logical input channels and proves that the public IO executes exactly the same
source schedule. The existing loop/execution/grid proofs are reused through
new actual-read specializations; their old endpoints remain short wrappers.
The new public storage domain requires signed bytes only in the finite FP8
input channels, and places no byte or finiteness condition on FP16 values.
`PublicLoadSafety`, `PublicStoreSafety`, `PublicKernelSafety` and `PublicSourceIO`
follow actual evaluated registers, loop successors and scheduled memories.
No access-safety or successful-execution premise is deferred as a numerical law.

The mathematical target remains independent RMS projection plus optional rotary
transformation. Native observation retains the original scalar, conversion,
reduction, K8 instruction and selected compiler profile. The input terms,
runtime dimensions, signed start, flags, aliases and complete grid remain in
scope. This format revision adds no numerical experiment requirement and does
not claim native arithmetic equals Real or introduces a second optimized kernel.

Optional proof layout (2026-10-21): `Misc/NativeCorrect.lean`, `Misc/Audit.lean`, `Misc/TensorBridge.lean` are optional entries in `Misc/`. The required `Contract.lean`, `Correct.lean`, `FPEquiv.lean` and their proof helpers do not import these entries. Declaration names, statements, proof terms and acceptance status are retained; this move adds no mathematical completion or numerical experiment.

Additional optional reader paths: `Misc/RawEndpoints.lean`. Their declaration bodies are retained. Historical readers remain separate from current primary/auxiliary checks.
