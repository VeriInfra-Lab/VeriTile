# Checked direct packed context attention

`context_attn_nopad_repaired.py` preserves the original production kernel's
floating expressions and widens index-producing integer expressions to int64.
The wrapper updates the supplied output object directly and returns `None`.
Only sequence metadata is copied to the CPU for validation.

The explicit contract is ordinary same-device CUDA FP32 Q/K/V/Out with the
same `[T,H,D]` shape and canonical contiguous strides. D is 16, 32, 64 or 128.
Start and length are contiguous int32 vectors. Negative/conjugate view flags
must be resolved. Read roles may alias. The output's full backing allocation
must be disjoint from each read role, including separate storage objects that
cover overlapping addresses. Actual nonempty data pointers are four-byte
aligned; storage sizes and formed pointers meet the signed-int64 byte ABI.

The wrapper checks nonnegative start/length, capacity, disjoint positive-length
packed intervals, a nonnegative built-in-int requested maximum, and native
launch-axis limits. It computes the actual maximum from the checked metadata.
Empty calls skip launch. Otherwise it launches the original `(B,H,ceil(max/128))`
grid with BM=BN=128, four/eight warps and one stage, inside the owning-device
context; the caller's device context is restored on return.

`Kernels.lean` exposes the typed operational source. `Proofs/HostPrograms.lean`
preserves the historical `hostProgram` and `wideHostProgram` wrappers and their
aliases; both public correctness entries export this helper. The composed
host-return proof dependencies are separate from the canonical source entry.

`Correct.lean` exposes `Checked.whole_mathematical_real`: the complete source
grid equals an independent strict-causal natural-exponential weighted sum over
arbitrary Real Q/K/V arrays, with no Real encoder restriction. This kernel uses
negative infinity for the causal mask. The theorem also preserves every input
region and every cell outside the output write footprint, and returns `None`.
It retains the actual ordered binary64 host scale `1.0 / (D ** 0.5)`.

`FPEquiv.lean` exposes `CheckedTensor.whole_tensor_fp32`: the supplied Tensor's
logical output words equal `Bytes.referenceWord`, derived from a separate
normalized online-softmax recurrence in `FPReference.lean`. The independent
reference explicitly specifies query/key/value tiles, max and exp order,
normalizers, casts, seeded dots, and the final accumulator; it does not invoke
the source executor. Checked prelude/body/epilogue/grid proofs connect the
actual typed source to that recurrence. Noncircularity and standard-base axiom
audits are attached to the public targets and theorems.

The canonical FP theorem also includes `Returned.MachineFacts`: actual native
control and active-address projection, exact int64 arithmetic for all formed
query/key/value/output lanes including masked lanes, queried source strides,
and actual four-byte role alignment. The same original bytes provide input
leaves, source metadata, and independent reference inputs. `logical_source`
and `logical_output` bind the shared Torch Tensor interface to those bytes.
Full output-byte readback, input backing windows, untouched bytes, empty calls,
output identity, `None`, launch device, and caller device are all constructed.

The selected native plan preserves SM80 TF32 `m16n8k8` instruction chains,
live dot accumulators, FMA contractions, rounded products, ordered physical
reductions, and natural-exp lowering through exp2. `NativeInstructions` exposes
only eight scalar instruction families and one fixed-k8 single-coordinate MMA
family as numerical ports. `Returned.whole_if_local` transports the entire
result under those nine all-word instruction laws. No whole-dot, reduction,
accumulator, final-output, or native-to-Real equality is assumed.

Ordinary Tensor/storage and CPU metadata observations, b32 byte transport,
compiler realization of the selected instruction plan, and successful
race-free launch delivery remain explicit boundaries. The shared launcher
product-overflow issue is tracked separately in `bench/TODO.md`; these proofs
do not establish runtime applicability for affected grids. Historical wide
metadata, staging, and copyback helper modules are retained as private reusable
proofs; the canonical checked entries and repaired Python use the direct
contract above. Existing original Python and original Lean APIs are unchanged.

`check_contract.py` checks exact original floating-AST retention and executes
the actual repaired kernel in Triton's CPU interpreter. Its CUDA metadata and
device context are explicitly facades; this is not GPU validation. It covers
all admitted dimensions, a multi-tile tail, packed gaps and backing frames,
empty calls, actual read aliasing, causal masking at very negative scores,
device restoration, invalid metadata/layout/dtype/flags, physically overlapping
storage objects, and misaligned pointers with byte remainders 1, 2 and 3.

The selected kernel readers use `KernelOnly/IO.lean` as their canonical IO
data module. It keeps the typed native `IOData` channels and the flattened
Real `RealData` view with their existing names; the distinct carriers and
source observations remain explicit. `KernelOnly/Correct.lean` only
instantiates that Real view. Input-binding, safety and execution connections
remain in `Proofs/IOData.lean`, `Proofs/RealData.lean` and the other proof helpers.
