# Finite-precision row-block correctness

The public `FPCorrect.lean` entry covers the entire previously accepted
per-program kernel body for fp32 input and output storage: both HAS_SCALE
branches, two-dimensional program ids, positive block size, partial masked
blocks and all-masked out-of-grid blocks. The independent value specification
keeps every padded -inf lane in the actual maximum, subtraction, exp, sum,
log and addition graph. It proves successful execution and exactly one memory
update, hence output readback and preservation of every other tagged cell.

The type-erasure bridge checks the typed body against the original accepted
source transcription. No real-number log-sum-exp identity, exp law, reduction
reassociation or all-masked-to-zero rule is assumed.

The binary32 primitive profile supplies actual ScalarOps add/sub/mul,
bit-preserving fp32 loads, literal 1 and -infinity, and an ascending-index
left-fold sum from positive zero. The software scalar profile selects scale=1
and proves both flag branches, retaining multiplication when the flag is true.
The general backend contract permits arbitrary scale. Maximum reduction,
exp and log are explicit word-level callbacks; every result is independent of
unused rest callbacks. A successful nonempty instance has one active lane
and one padded lane, with exact output and memory frame.

All three public declarations are backend-value contracts. There is no closed
bit interpreter for the transcendental primitives and no GPU correctness
claim. Native binding must establish fp32 storage and pointer representation,
scalar rounding/FTZ policy and FMA contraction, maximum reduction order plus
NaN/tie/signed-zero behavior, exp and log instruction/library semantics, and
the sum tree. The sequential sum profile is not claimed to equal the generated
GPU tree. NaNs, infinities and all-masked results follow these callbacks;
they are not replaced with the old real model's finite fallback.

The old per-program scope is fully covered for this explicit fp32 profile.
Host launch assembly, host `z.logsumexp(-1)`, output dtype conversion and other
input dtype profiles remain outside that accepted kernel scope.
