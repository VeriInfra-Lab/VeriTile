# Kernel revision: checked FP32 context attention

This revision adds kernel-only statements while preserving the selected Python,
all guards, the contract, and all historical theorem bodies. The four standard
entry modules are `Kernels`, `Contract`, `Correct`, and `FPEquiv`.
`PublicKernel.correct_real`, `correct_fp32`, `shared_source`, and `input_tensor`
are public aliases. The older README describes the preceding delivery; the
new compilation and audit evidence below belongs to this revision.

## Independent targets and precision

Q and Out are FP32 `[T,H,D]`; independent K and V caches are FP32
`[B,Hkv,Sk,D]` and `[B,Hkv,Sv,D]`. D is one of 16, 32, 64, 128, 256.
The actual source retains its ordered QK dot, finite causal sentinel `-1e8`,
base-2 exponentials, sum, explicit old-normalizer update, old-output rescale,
FP32 probability cast, live-accumulator PV dot, and final division.

The launch expression is Python's ordered
`1.0 / (D ** 0.5) * 1.4426950408889634`, passed through the FP32 scalar ABI.
`KernelScale.value` fixes the exact incoming scalar, separately from payloads:

| D | FP32 bits | exact Real value |
|---|---|---|
| 16 | `0x3eb8aa3b` | 12102203 / 33554432 |
| 32 | `0x3e8293ee` | 4278775 / 16777216 |
| 64 | `0x3e38aa3b` | 12102203 / 67108864 |
| 128 | `0x3e0293ee` | 4278775 / 33554432 |
| 256 | `0x3db8aa3b` | 12102203 / 134217728 |

`KernelScale.Bound` is an input-scalar ABI observation. The older abstract
HostArithmetic interface is preserved; it is not used to claim an in-Lean
proof of CPython binary64 arithmetic or runtime FP32 conversion. The five
entries were checked against the actual ordered Python expression and pack
conversion. They do not constrain Q/K/V words or assert a numerical result.

For local query row i, let
`E=min((floor(i/BM)+1)*BM+prompt,sequence)` and `J=ceil(E/BN)*BN`.
`KernelOnly.mathematical` is the independent initial-array formula
`sum(j<J, 2^score(j)*V(j)) / sum(j<J, 2^score(j))`, using the grouped KV head,
zero V beyond E, scaled QK for `j <= i+prompt`, and finite `-1e8` otherwise.
The padded/future terms retain positive Real weights. This is the exact
source formula; arbitrary Real inputs do not give strict causal softmax.
Its theorem concludes successful complete flattened ordinary Real source
execution, every active result, full frame, read preservation, and logical
coverage. Real arrays require no finite FP32 encoding.

`KernelOnly.nativeTarget` starts from original bytes and the independent
`FPReference` online recurrence, then evaluates the fixed native graph.
It never selects a successful source result. `native_tensor` concludes actual
shared-source grid success, equality with the independent reference memory,
exact bits in the original output Tensor, every active byte, exterior-byte
frame, all read windows, tagged-memory frame, and empty-call frame. Original
Q/K/V Tensor values are bound by the separate `input_tensor` theorem.
No finite-value restriction, native-to-Real equality, or error bound is added.

## Domain, grid, and original storage

The unchanged checker requires ordinary CUDA Tensor/storage observations,
FP32 data and int32 contiguous metadata on one device, dense Q/Out and dense
K/V inner axes, legal grouped heads and widths, positive heads, legal packed
intervals, independent K/V capacity bounds, resolved flags, aligned data
pointers, and finite storage windows. K/V batch gaps and nonzero storage
offsets are supported. Read-only inputs may alias. The full output backing
window is separated from every read window; positive output intervals do not
overlap. Formed offsets, including masked lanes, fit signed int64.

BM=BN=64 for the existing Tesla-name branch and 128 otherwise; warps are 4
for D<=64 and 8 otherwise, with one stage. The grid is
`(ceil(max_input_len/BM), B*H, 1)`, with the existing `B*H<=65535` check.
No payload copy or private output is introduced. Kernel statements omit the
host return/device result while retaining the original object and byte
bindings. Existing host-return/device theorems remain available unchanged.

## Native realization and evidence boundary

The selected profile is CUDA SM90, Triton 3.7.1, default TF32 input precision,
and fixed m64/n16,n32,n64,n128,n256/k8 WGMMA instructions. QK clears its first
instruction; later K8 instructions carry the previous result. Every PV
instruction uses a live seed, including a numerically zero seed. The first
PV seed is the already rescaled output. The nine scalar/instruction ports
contain no whole dot, reduction, accumulator, or final-output oracle.

The ten actual-wrapper/native-binder configurations (both BM branches and all
D values) compile offline. Static symbolic PTX checks compare exact local
left-fold trees and XOR2/XOR1 reduction replicas, scale operand order,
normalizer FMA order, old-output multiply order, clear/live bits, K8 coverage,
every PV register-input coordinate/producer, stored owners, and division
normalizer ownership. The audit operates on labels and syntax, not floating
values. `KernelProfile` additionally proves routing, operand order, and
clear/live selection independent of raw words and arbitrary Real literal
fields. The actual source CPU checker covers execution and contract guards;
it is not an instruction accuracy experiment. No GPU was launched.

Ordinary original-storage observations, b32 transport and instruction-local
semantics, compiler realization beyond the checked profiles, and successful
race-free launch delivery remain explicit applicability boundaries. Offline
PTX compilation does not establish launchability: BM128/D256 reports 262144
shared bytes, above even the SM90 233472-byte per-SM shared-memory capacity. The shared
launcher's already tracked grid-product overflow is also outside this source
proof; no new grid guard is claimed. The packet does not assert universal
runtime success on every contract-valid device/shape.

## Build and preservation

All 48 current task Lean modules were rebuilt in the workspace overlay with
the reviewed targeted driver, using current shared dependencies where needed.
This is a task rebuild, not a cold rebuild of the transitive repository.
Receipts, effective object hashes, source import closure, available ilean
imports, and missing reused import artifacts are recorded. The prior 48
files are archived before revision. Only Correct/FPEquiv receive new imports
and aliases; their historical content remains present, and all other old
source hashes are unchanged. No core, Lake, ledger, or frozen prior packet
is modified. New evidence must receive independent review before acceptance.
