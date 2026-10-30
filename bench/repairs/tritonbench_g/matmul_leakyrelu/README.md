# Checked matmul_leakyrelu contract

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `KernelOnly/Misc/TensorBridge.lean`, `KernelOnly/Misc/NativeCorrect.lean`, `KernelOnly/Misc/PublicAudit.lean`.

`matmul_leakyrelu_repaired.py` adds input admission checks and a post-allocation output
byte-envelope check before the original launch. Its supported inputs are rank-two strided FP32 A/B on the same
CUDA device, compatible dimensions, contiguous A, resolved negative views,
nonnegative strides, a builtin-string activation, and the original
`Whole.IndexDomain` signed32 envelopes. All formed padded indices and the final
K-pointer advance are covered. Nonempty FP32 A/B data pointers are checked for
natural four-byte alignment. `OrdinaryTensorContract.admitted` derives both
input-alignment fields from these checks and the actual pointer observations.
Exact input byte-envelope checks cover padded/final A/B pointers below uint64.
The allocated C pointer receives the corresponding check before launch; this
can fail after allocation, with no kernel stores. `OrdinaryTensorContract` has
no padded-address premises. Its three legacy byte bounds are derived from the
input checks and the explicit successful `outputCheck` setup condition.
Ordinary Tensor bindings retain valid byte spans and fresh FP16 output allocation.
The output allocation and actual launch both use `a.device`; the launch context
restores the caller's prior device, so no current-device equality is imposed.

B keeps its supported nonnegative transposed/sliced/broadcast strides and
storage offsets. No copy or B-contiguity check is introduced. Inputs may alias
consistently. All strings are accepted: only `"leaky_relu"` activates the branch.
The fixed 32×32×32 blocks, group size4, all ceil(K/32) blocks, ragged tails,
short final groups, K=0 and M/N empty-grid cases remain unchanged. NaN and Inf
inputs are not silently excluded.

The mathematical formula is the finite-K matrix product, optional exact
1/100 leaky slope, and explicit `roundHalf`. The selected floating formula is
an independent logical recurrence with a live-accumulator default TF32 dot,
right-hand slope multiplication when activated, and final FP16 conversion.
The accepted offline SM90/Triton3.7.1 graph and its source correspondence are
retained; their detailed scope is documented in the original task README.
The source's separate-dot/add graph and the observed contracted graph are
provably different on floating inputs. No equality between those graphs is
asserted or deferred.

`Proofs/Contract.lean` proves exact admission equivalence and transports the
checked envelope to every activation string. `Correct.checked_mathematical`
and `FPEquiv.checked_floating` reuse the accepted complete contracted host/grid
proofs, including each FP16 output, exact full frame, and source safety. Six
local numerical families remain explicit: initial zero, masked zero,
live-accumulator dot, ordered comparison, right-side slope product, and FP16
conversion. GPU numerical conformance remains deferred.

All 14 accepted closure modules and five checked modules compile with axiom
audits. `python bench/repairs/tritonbench_g/matmul_leakyrelu/check_contract.py`
passes unchanged-kernel/config AST checks and verifies the new output guard, 12 admission families across four activation
strings, 19 metadata/index/alignment/address rejections and three non-string activations, exact
allocation/launch/return checks, propagated failures and actual CPU negative-view
observations. CUDA is mocked; no GPU run or historical acceptance change is
claimed.

### Launch-device admission correction (2026-10-19)

Prior candidate SHA-256: `0240070b2db13874a278b632f241768ab223a917b314e3d26e71b86f6487a18d`.
The installed Triton JIT chooses the current CUDA device and stream; equal
input devices alone did not select the owning tensor device. The actual source
submission now enters `torch.cuda.device(a.device)` and restores the
ambient caller device. Formal host device/stream observations refer to this
effective context. CPU controls check a cuda:1 launch from an ambient cuda:0
caller and restoration on success and submission exception. Kernel AST and
numerical semantics are unchanged. The prior hash is retained as acceptance
correction evidence; no native GPU/runtime conformance is claimed.
