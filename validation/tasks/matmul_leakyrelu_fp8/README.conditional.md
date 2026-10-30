# Selected FP32 matmul with optional leaky-ReLU repair

Read Kernels.lean, Correct.lean and FPEquiv.lean. The production wrapper has no
input dtype conversion or dispatch: source_specified_else_fp32 selects FP32
A/B, with supplied FP32 dot accumulator and unconditional FP16 output. The
filename does not establish an FP8 route. The sole configuration is
BM128/BN256/BK64/GROUP8/stages2/warps8. Exactly "leaky_relu" enables the original
where(x>=0,x,0.01*x); all other strings use identity. Final cast follows activation.

This separate candidate repairs int64 arithmetic, quotient/remainder ceilings,
shortened group span, masked K/C pointer formation and final K overshoot. It
makes raw/resolved admission explicit and adds fresh-empty and K0 zero paths.
Original modulo loads, nested local PID remainder, K order and fusion remain.
SourceCorrespondence proves group-coordinate preservation and equality to the
original independent matrix-product/activation target.

The full domain is ordinary rank-two FP32 tensors on one CUDA device,
compatible K, contiguous A and nonnegative strided B, including offsets,
broadcast and read-only A/B aliasing. Stored negative values are allowed;
lazy neg/conj flags are rejected. Empty-view offsets need not be dereferenceable.
Fresh C has shape M by N, FP16 dtype, offset 0 and stride (max(N,1),1).
All M/N/K branches and every caller allocation are covered. Allocation and
ordinary compiler/runtime completion stay explicit implementation boundaries.

Bounds/ControlBounds prove actual integer and formed-pointer limits.
Execution/OutputCoverage/GridOrder establish all evolving-memory grid results.
Autotune proves idempotence and finite repeated-launch equivalence. Host executes
compatibility, allocation, empty/K0 branches, trial sequence and exact return;
HostReal and HostFrame assemble the independent formula and complete frame.

FPEquiv uses the same actual artifact/AST/ABI/tape with current reaching IDs.
Its four local numerical fields are cast, comparison, ordered slope multiply,
and one fixed SM90 WGMMA TF32 instruction. DotPlans covers all BK64 lanes by
eight K8 fragments. Five cases include both native multiply operand orders;
the reference retains source left multiplication. Tape.Admitted is explicit.
CompilerCorrect refines the complete custom source interpreter; generic typed
DSL translation, unobserved ISA mappings and native-to-Real accuracy are not claimed.

26 owned Lean modules include Reader (25 library modules). Reader applies both
whole-task public statements, audits axioms/circularity and prints all four
local fields plus the domain premise. 30 production-AST CPU cases, 8 real Torch
metadata/host cases and 4 manual ASTSource SM90 compile profiles passed, with
separate original logical-view witnesses and candidate rejections. They do
not establish GPU execution or native numerical conformance.

Original SHA256:
5d8796c8b490469195f1e25d4807e5a0a214ea65e4481e1b34075ce4b377fa55.
Candidate SHA256:
0aadc6c61e44ea0e3c08fa532887ecd79d8b0b89ab351bf7bb756c8a599ce3bc.
Independent final review and root integration remain separate from producer
builds. Original source/proofs/status are preserved; acceptance increment is zero.

Resource revision002: the original sole pipeline stages=3 configuration and
first candidate emitted 294912 bytes (288 KiB) of shared memory on all four
selected SM90 profiles, exceeding the CUDA SM90 total 228 KiB limit. The
repair explicitly changes stages from 3 to 2; tile constants, warps, source
K-loop/dot/activation/cast ordering remain. Four fresh offline profiles now
emit 196608 bytes (192 KiB), removing that concrete resource obstruction.
This is an explicit configuration repair, not a numerical assumption or a
claim of actual GPU admission. First-version proofs and controls stay frozen.
The 30 CPU and 8 Torch controls are reused only after exact equality of their
executed ASTs is checked; their historical stages3 CPU annotations remain
labelled as such. Neither CPU control is rerun. New native configuration
evidence is the stages2 offline compilation, not that AST transfer.
