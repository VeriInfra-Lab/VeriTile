# Checked `triton_matmul` candidate

`triton_matmul_repaired.py` is a separate candidate. The benchmark Python file
at `bench/tritonbench_g/triton_matmul/triton_matmul.py` is unchanged (SHA-256
`5d64fb3c49162e09db20cd61dd720c2fefbbf4fadbcc20ecc7d51873431dc004`).
The selected `KernelOnly/Correct.lean` entry exposes `KernelOnly.kernel_real`
for independent Real mathematics; `KernelOnly/FPEquiv.lean` proves the complete original/shifted source pair. `KernelOnly/Misc/NativeCorrect.lean` retains the ordinary
`KernelOnly.kernel_native_if_local` observation for native storage words. Both
Real and native results use shared typed `ProgramIO` over finite logical
matrices, original strides and the complete scheduling permutation, proving
source access safety, successful execution, outputs and the exterior cell frame. `KernelOnly/Misc/TensorBridge.lean` retains
the complete Tensor, finite-grid and physical-byte applications.

## Wrapper and source scope

The checked wrapper accepts matching FP16 or E4M3FN, rank-two, default-contiguous
strided tensors on the same CUDA device, with compatible inner dimensions and
resolved negative flags. It passes the original input objects without copies.
Singleton unused strides and contiguous storage offsets are allowed. It allocates
and returns a fresh same-dtype C on that device, with an explicit empty-output
branch that performs no launch or input-pointer query.

Every dimension is below `2^63`. For nonempty output the wrapper also checks

* `max(width*M*K, width*K*N, width*M*N) < 2^63`;
* `ceil(M/128)*ceil(N/256) < 2^31`;
* each nonempty input pointer is aligned to its element width and its byte-window
  end is at most `2^64`.

These are explicit source-index and pointer conditions. Ordinary tensor and
allocator laws describe storage metadata, contiguous indexing, output freshness,
and representable output storage; they do not assert a source arithmetic envelope
or the kernel result.

The original profiles and arithmetic are preserved:

| Storage | BM | BN | BK | Group M | Stages | Warps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 128 | 256 | 64 | 8 | 3 | 8 |
| E4M3FN | 128 | 256 | 128 | 8 | 4 | 8 |

The kernel retains the live FP32 seed in the three-argument `tl.dot` and the
explicit cast to its output storage dtype. There is no autotuner. Source repairs
widen before metadata/index arithmetic, compute ceilings by quotient/remainder,
clamp both coordinate addends before adding, recompute safe K pointers on each
iteration, remove the terminal pointer advance, and remove invalid contiguity
hints for clamped lanes. Canonical wrapper strides are used.

## Proof chain and trust boundary

Both original and candidate `Kernels.lean` import only source definitions.
`bench/tritonbench_g/triton_matmul/SourceDefinitions.lean` contains the extracted
historical source declarations; backend proofs remain under `Proofs`.

`Contract` proves exact guard/precondition equivalence. `TensorBinding` and
`InputAdapter` connect the original tensor objects, storage offsets, pointers,
and logical payloads to source memory. `FiniteExecution` executes signed64
controls, loads, K-loop arithmetic, masks and offsets, and refines the typed
source. `FiniteGrid` and `FiniteHost` compose every program, allocation, the empty
branch, returned-object identity, result, and exterior logical-memory frame.

`IdealTensor.expected` independently sums the full K dimension of arbitrary
Real-valued original logical inputs. Its physical memory reader has codomain ℝ,
with no finite-payload representability restriction. Ordinary tensor view laws
connect storage offsets and aliases. `IdealHost` and `Correct.matmul_checked_real`
compose the same checked allocation and actual finite grid into the full result
and caller frame. `Correct.matmul_checked` additionally retains the decoded
finite-payload bridge through `CheckedMath.expected`. Both ideal interpretations
are separate from floating-point rounding.

`Native` derives variable-shape dot behavior from fixed K16 FP16 or K32 E4M3FN
instruction groups with the live FP32 accumulator and padded lanes. The
independent `FPEquiv.expected` recurrence uses these groups and the final storage
cast. Its numerical premise has four local correspondence fields: the two MMA
instructions and the two output encodings. It does not assume a whole-kernel
result or identify native dot with sequential FMA.

`OutputBytes`, `ByteExecution`, and `ByteHost` execute the finite C pointer and
actual one- or two-byte stores, compose their simulation through the full grid,
and establish C readback through its ordinary tensor storage offset. The native
public theorem identifies the actual allocated/returned C and establishes every
output word, every output byte, and every exterior byte. Separate proved
corollaries establish output-address bounds and physical input-byte preservation.

The old broader helper entries are retained as `Proofs/HistoricalCorrect.lean`
and `Proofs/HistoricalFP.lean`; they are not the current wrapper contracts.
Shared launcher compatibility and historical package cleanup are separate work.

## Validation

* `check_contract.py`: 26 preallocation rejection controls and 18 interpreted
  calls, covering both dtypes, tails, group tails, K=0/empty output, storage
  offsets, singleton strides, aliases, returned allocation identity, input bytes
  and exterior sentinels. Synthetic metadata controls exercise the byte and PID
  admission boundaries. Source/guard checks preserve exact profiles and precision.
* `check_cuda_lowering.py`: four offline SM90 compilations on Triton 3.7.1,
  covering both storage profiles and i32/i64 scalar signatures. Audits observe
  fixed K16/K32 MMA, live FP32 accumulation, storage casts and signed64 indexing.
* CPU interpreter tests use sparse `-1`, `0`, `1` data with exactly representable
  outputs. Interpreter E4M3FN rounding differs from Torch on some ties; these
  checks establish addressing and storage controls, not general native numerics.
* CUDA execution is unavailable. The four local numerical correspondences remain
  explicit and unvalidated against a GPU in this environment.

The full local rebuild covers 45 distinct original/candidate modules: the
43-module dependency rebuild passed, followed by the two arbitrary-Real helpers
and a rebuild of the strengthened public entry. There were no errors. All public
entries include standard-base axiom audits and the independent specifications
include noncircularity checks.

The selected `KernelOnly/Correct.lean` facade exposes the mathematical specification. The optional `KernelOnly/Misc/NativeCorrect.lean` facade retains the ordinary auxiliary native theorem. `KernelOnly/FPEquiv.lean` proves the original/shifted source pair; optional native results require an explicit Misc import. Optional logical Tensor application and input-byte frame exports are now in `KernelOnly/Misc/TensorBridge.lean`, under `KernelOnly.TensorBridge`; their existing supporting proofs remain unchanged.

The KernelOnly developments retain `realIO` / `nativeIO`, independent
logical-matrix formulas and shared `ProgramIO.Implements`. FP16 and E4M3 input
and output channel types remain explicit, as do the native K16/K32 instruction
profile and the four original numerical correspondence premises. Real uses
the isolated ideal backend; no native=Real statement is introduced.

`KernelOnly/Proofs/SourceSafety.lean` follows the actual selected source prelude,
every masked K-loop load and successor context, and masked output stores.
`PublicIO.lean` proves its full schedule equals `runPrograms`, preserving program
IDs, num-program IDs and every original scheduling permutation. Finite matrix
bindings cover exactly the active inputs. `KernelOnly/Misc/Operational.lean` retains the former
raw-state statements as ordinary theorems for audit. Tensor metadata, physical
bindings, finite-grid execution, byte observations and Tensor-dependent
safety/empty helpers remain unchanged in `KernelOnly/Misc/TensorBridge.lean`.


## Program-pair review, 2026-10-21

`KernelOnly.FPEquiv.grid_equiv` proves the complete actual `original ≡[[]]
shifted` source pair. `KernelOnly/CandidateKernels.lean` is a complete ordinary
triton block. The sole delta replaces `GROUP_SIZE_M * num_pid_n` by
`num_pid_n << 3` when GROUP_SIZE_M=8, retaining the original formula otherwise.
The Python candidate is `KernelOnly/triton_matmul_shifted.py`; all wrapper
checks, both configurations, strides and launch flow are preserved.
This is integer address strength reduction, with no measured speed claim.

The proof covers arbitrary full grid permutations, FP16/E4M3 operand and output
storage, FP32 live accumulation, every K block, clamped tail coordinates,
safety, successful execution, all output cells and both complete exterior
frames. Inputs retain all word patterns; no finite restriction or numerical
rewrite rule is used. The primitive MMA/cast interpretation is identical on
both sides, with the original fixed instruction interfaces. `Proofs/Pair.lean`
proves exact statement evaluation/safety and lifts through the full source and
grid. `KernelOnly/check_pair_source.py` verifies exact complete-module AST
restoration of only this guarded shift and names.

One complete typed-source Implements/native recurrence remains available as an optional observation, under local fixed instruction/encoding correspondences. These computational model boundaries are valid and not themselves format defects; there is no second comparable implementation or schedule. Repair-vs-original tail candidate is separately documented as pending/defective and cannot be substituted as an equivalent pair.


## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.


## Literal function-body migration

The selected Python source remains `triton_matmul_repaired.py`, function
`matmul_kernel_repaired`; the comparison source remains
`KernelOnly/triton_matmul_shifted.py`, function `matmul_kernel_shifted`.
Neither Python module is edited by this migration. Each of their 29 top-level
statements is retained in each FP16/E4M3 `triton { ... }` body. Pointer and scalar
dtypes belong to the external `CallSignature`; no `tl.region` declaration is
inserted into the pasted body. The source `launch_metadata` callback is recorded
by the correspondence checker, not interpreted as a numerical proof premise.

The bodies retain the simultaneous metadata cast assignments, every
`.to(tl.int64)`, quotient/remainder ceilings with the Boolean-to-int64 casts,
the computed `blocks_k`, the signed loop, live FP32 dot accumulator, and both
branches of the storage-dtype-dependent final conversion. The candidate's only
change remains its guarded group-width shift and entry names. The complete-module
AST restoration check continues to check the surrounding Python code as well.

`Proofs/RawPrelude*`, `RawBody*`, `RawLoop`, and `RawMemory` evaluate the actual
copied source. `RawCoordinates`, `RawModel`, `RawLaunch`, and `RawWritten` connect
its signed coordinates, masks, pointers and computed loop count to the existing
independent matrix recurrence. The source does not use the old independent
`numKBlocks` field: the host-derived value is proved equal to the computed count.
`RawInactive` proves that ids beyond the row groups write no output, retaining
the generic host helpers' domain. The bridge equates final memory, not temporary
registers or syntax trees. It assumes no floating identity or complete execution
result.

`RawSafety` and `RawSafetyLoop` follow the actual signed loop and its successor
states. The public safety proof derives the needed coordinate facts from the
existing legal grid condition. `KernelOnly/Proofs/SourcePair.lean` proves the
actual original/candidate statement and trace equivalence; `Pair.lean` lifts it
through the unchanged public schedule. The selected Real and FP declarations,
complete grid permutations, storage profiles, zero-axis behavior, input strides,
and independent mathematical specifications retain their public domains.

The historical `LogicalViewDefect` witness now constructs its own explicit
Real algebra instead of referring to a separately revised FP backend with an
incompatible carrier type. Its aligned +64 versus logical -64 counterexample
and the downstream Real bridge are retained; this is not a native FP claim.

Current migration verification is recorded separately from the historical
interpreter/offline-lowering results above. This environment has no Torch/Triton
runtime or usable official comparator isolation, so this revision claims no new
GPU execution, numerical correspondence measurement, or performance result.
