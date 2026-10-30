# Parallel retention attention

The original selected FP32 Python kernels and wrapper are retained. Separately named actual candidates fuse the adjacent accumulator multiply and add assignments in the forward first loop and DQ first loop. For example, `b_o = b_o * decay; b_o = b_o + dot(...)` becomes `b_o = b_o * decay + dot(...)`. DQ uses the corresponding `b_dq` fusion; the complete backward parent calls the candidate DQ, retains the barrier, and reuses original DKV. This removes intermediate same-name assignments without changing the ordered primitive graph; no measured speedup is claimed.

`SourceSyntax.lean` and `QuerySyntax.lean` contain the readable original quotations, with compatibility imports at the old proof paths. `CandidateSyntax.lean` contains direct candidate quotations. `ShellSyntax.lean` displays both complete backward bodies. `Fused.shell_body`, `fusedShell_body` and `shell_original` prove their connection to the retained actual DQ/barrier/DKV source. The source precision, explicit FP32 load/cast boundaries, signed64 diagonal differences, eager masks, both loop phases and descending DKV order remain intact.

`FPEquiv.forward_fused_equiv` and `backward_fused_equiv` are the main actual two-program conclusions, with `R=[]`. They retain every FP32 word, all valid original per-program tiles, every output shard, successful source execution, access safety and both exterior tagged-cell frames. The exact proofs work for the retained arbitrary scalar/dot-tree interpretation; they use no numerical rewrite identity or experiment. The backward pair also holds without the positive-width premises needed by the optional differentiated-output formula, so its domain is no narrower. Original independent Real specifications and optional native/Tensor applications are unchanged.

`Contract.lean`, `Kernels.lean`, `Correct.lean`, and `FPEquiv.lean` are the
kernel-facing entry points. Historical host sum(0), saved-context, returned-API,
and VJP proofs remain under `Proofs/`; all 41 preexisting files also remain
byte-for-byte in the initial preservation snapshot. Public kernel theorems
quantify every valid launch tile and every active scratch coordinate. They do
not add a host, allocator, compiler, or byte-heap subsystem.

## Contract and independent targets

Q/K are `[B,H,T,DK]`, V/G are `[B,H,T,DV]`, and the checked variant accepts FP32
CUDA tensors. The original contiguous materializations remain. DK/DV are
positive; empty B/H/T follows the existing no-launch wrapper path. Tile widths
are min(128,max(16,next_power_of_2(D))). Explicit zero padding supports both
sequence and feature tails. The wrapper asserts shape, dtype, device, view
flags, signed32 coordinates, signed64 formed offsets, pointer room and
individual CUDA grid axes. Fresh scratch buffers supply ordinary separation.
Grid-product launcher compatibility remains the explicitly deferred runtime
issue; no new grid-product restriction was introduced.

The mathematical full output is
`O[t,v] = DK^(-1/2) sum_{n<=t} gamma[h]^(t-n) (sum_e Q[t,e] K[n,e]) V[n,v]`,
where `gamma[h] = 1 - 2^(-5-h)`. `Proofs/KernelRealTarget.lean` states its
independent feature-shard contributions and the three differentiated causal
contributions. `Correct.forward` and `Correct.backward` expose shared
`ProgramIO` records and prove that the source implements those independent
formulas. The records show Q/K/V/G channels, dense input addresses, output
shards, active masks, the actual source invocation, and no private scratch.
The shared judgment includes actual source trace safety, successful execution,
typed output cells and the complete exterior cell frame. The native file
imports no task Real target or correctness proof.

`Proofs/KernelWordTarget.lean` separately states the source-precision FP32
instruction graphs over entry-memory words. The optional `FPEquiv.forward` and
`FPEquiv.backward` theorems in `Misc/NativeCorrect.lean` expose the same source IO with every FP32 word admitted and
the existing native scalar/dot-tree observer; their independent formulas retain
the original operation order. These are source-to-native observation results,
without a second optimized kernel or a claim that FP32 arithmetic equals Real
arithmetic. `TensorBridge.forward` and
`TensorBridge.backward` retain successful execution and every byte of O/dQ/dK/dV
scratch Tensors at their actual data pointers, with allocation bounds and the
exterior byte frame. Scratch metadata has the original leading shard dimension;
the source wrapper's subsequent sum(0) remains in the retained historical API
proofs. Entry metadata binds the prepared Q/K/V and incoming cotangent pointers.

## Proof path and model boundaries

`KernelForwardResult`, `KernelQueryResult`, and `KernelBackwardResult` connect
all source loop phases to the independently stated instruction graphs.
`KernelProgramMemory` proves source success and masked store readback.
Uniqueness is required only among effective lanes: padded lanes may share an
integer address and issue no store. `KernelPhysical` realizes those completed
source stores using the existing one-word `LocalRuntime` law and proves the
word frames. The existing little-endian `NativeBytes.runtime` supplies concrete
four-byte stores, decoding, and byte frames. No final result, successful run,
whole compiler-conformance, or whole-output equation is an input premise.

FP32 scalar conversions/constants, add/sub/mul/FMA, log2/exp2, and host scale
conversion are explicit elementary primitive models. Finite dot trees contain
instruction syntax and a coverage proof. The task-local dot-to-dotAcc staging
passes the live accumulator into that tree, preserving FMA order; it asserts no
split-add numerical identity. Same-format FP32 loads/casts preserve words.
The selected source retains `allow_tf32=False`, source casts, scale placement,
and operation order. These are primitive/compiler model boundaries, with no
FP32-equals-Real claim and no numerical-experiment condition on the theorems.

The source adapter uses the existing natural-coordinate interpreter under the
proved integer guards. The backward shell's pure PID arithmetic is expanded
at its helper-call boundary. Its descending time range is reindexed by a
nonnegative loop counter with `descendingCount`/`descendingStart`; the loop
proofs retain the same order. The wrapper and compiler internals are ordinary
external models, not newly verified subsystems.

## Bounded validation

- `check_quotation.py` compares all three complete Python/DSL bodies and the
  backward shell's exact PID/call structure. It records only explicit syntax,
  checked integer cast/promotion, and reverse-loop normalizations; it performs
  no floating arithmetic reassociation.
- `check_precision.py` checks pinned Python/AST hashes, unchanged original copy
  decorator, separate Real/native imports, and retained eight actual-JIT
  SM80/SM90 profiles with FP32 SIMT dots and live carries.
- Scoped Lean 4.29.0 builds check the public endpoints, source and readback
  proofs, standard axiom footprints, and independent-target dependency audits.
- The preserved CPU interpreter/autograd controls and offline compiler
  artifacts remain evidence for their recorded profiles. No GPU execution is
  claimed, and the theorem statements do not depend on a GPU experiment.

The frozen task packet records source hashes, exact build receipts, bounded
checks, preserved files, and the remaining model boundaries for peer review.

The primary `Correct.forward`/`backward` and optional `FPEquiv.forward`/`backward`
theorems in `Misc/NativeCorrect.lean` contain IO, independent formulas, dimensions, valid program indices
and the existing primitive models. Operational states and physical transport
details are ordinary theorems in `Proofs/`. `PublicIOBounds` derives access
bounds from the declared windows; `PublicAccessForward`, `PublicAccessQuery`
and `PublicAccessBackward` prove safety through the actual evaluated loops.
`PublicForwardIO` and `PublicBackwardIO` compose source execution, input binding,
all output values and exact frames without assuming successful execution.

`Proofs/PhysicalEndpoints.lean` preserves the exact previous Real and native
physical endpoint statements. `PublicPhysicalTransport` explicitly binds
physical input words to the public channels and proves that each physical run
is the public source schedule followed by the original masked commits.
Input bases may alias; only the original backward output separation is used.
`Misc/TensorBridge.lean` retains its complete original theorem statements, including
every output byte, finite address bounds and the full exterior byte frame.
The public scope remains one complete forward program or the complete combined
backward shell, for every valid tile; no full-grid or host-sum claim is added.

## Selected pair review (2026-10-21)

Missing pair target: changing primitive interpretations of the same API is not an implementation transformation.
The previous single-source native/software results now live in
`Misc/NativeCorrect.lean` as ordinary theorems. Their fully qualified names, complete
statements, proof bodies and input domains are retained when importing
`Misc/NativeCorrect.lean` explicitly. They are useful auxiliary correctness results and do
not count as a completed two-implementation FP equivalence. The independent
mathematical `Correct` target is unchanged.

Optional native observations, Tensor/host applications and reader/audit entries
are in `Misc/`. Import these modules explicitly when needed; the primary
`Correct.lean` and `FPEquiv.lean` entries do not re-export optional results.

The current assignment-fusion checks include exact original/candidate quotation
binding and the actual candidates under the existing CPU interpreter suite
(6 full forward/backward cases, 37 host admissions and 21 rejections). CUDA
metadata/context are shimmed as documented by the existing checker. No GPU
execution, candidate compiler admission or performance measurement is claimed.
Historical compiler receipts remain bound to the unchanged original source
definitions; the precision checker distinguishes their original source prefix
from appended separately named candidates.
