# Checked FP32 matmul with optional leaky-ReLU

The checked entry is `matmul_leakyrelu_fp8_repaired.py::matmul`. Its public
statements are `Checked.correct`, `Checked.fp_equiv`, and `Checked.frame` in
`Correct.lean` and `FPEquiv.lean`, under the namespace
`VeriTile.Bench.Repairs.MatmulLeakyreluFp8`. The original benchmark and the
historical conditional candidate/proofs are preserved separately.

## Source and contract

The source has no input conversion or dtype dispatch: inputs are FP32, the
supplied dot accumulator is FP32, and output is explicitly cast to FP16.
The filename does not establish an FP8 route. There is no source AMP branch;
the historical FP16 fixture does not add an input profile. Exactly
`"leaky_relu"` selects `where(x >= 0, x, 0.01 * x)` before the half cast;
every other string selects identity. No numerical-value guard is imposed.

Admission requires actual rank-two strided FP32 tensors on the same CUDA
device, compatible inner dimensions, contiguous A, nonnegative strides,
resolved neg/conj flags, aligned active FP32 pointers, and a string activation.
The existing B domain includes transposes, offsets and broadcast strides.
A/B may alias read-only. Ordinary Torch storage bounds and signed descriptor
ranges remain metadata laws. Empty views need not be dereferenceable.

The wrapper validates before effects, calls `torch.empty` with the actual
shape/device/FP16 arguments, returns that allocation directly for empty
outputs, and uses `zero_()` for K=0. Positive work launches inside the input
owner's CUDA context. The actual allocated object is returned, with shape
M by N, offset zero and strides `(max(N, 1), 1)`.

The sole configuration is BM128/BN256/BK64/GROUP8/stages2/warps8. The checked
kernel AST is identical to the preserved conditional candidate. It retains
the historical int64/control/pointer repairs, shortened groups, safe masked
pointers, source K order, activation/cast order and default dot arithmetic.
Stages2 is the historical explicit resource repair: the former stages3
configuration emitted 294912 shared-memory bytes on four offline SM90
profiles; stages2 emitted 196608. That historical compilation evidence is
not a claim of actual GPU execution or universal runtime admission.

## Checked composition

`Contract.lean` proves accepted-domain equivalence for the grouped metadata
checker, including unsupported raw input classes. Raw Python descriptor
extraction and exact exception-label ordering are not formalized. The
pinned Python/Triton-to-event correspondence is a manual interpretation
boundary, supported by executable controls.

`Proofs/Checked/Events.lean` exposes scalar memory, cast, comparison and slope
events plus one fixed K8 instruction result lane. Eight ordered K8 events
form each BK64 block. `Source.lean`, `Replay.lean`, `Reference.lean` and
`Real.lean` prove all K iterations, masks, grid cells, stores, zero-fill,
arbitrary finite autotuner repetitions, the independent caller-coordinate
floating-point recurrence and the independent dense Real formula.

`Host.lean` and `Success.lean` compose validation, actual allocation, consumed
launch owner state, computation, return identity and successful context
restoration. The ordinary local runtime/storage interpretation remains
explicit. There is no whole-call compiler-correctness premise in the new
public path. `Execution.lean` derives signed64 element arithmetic and active
unsigned64 byte-pointer bounds, including the four bytes of offset/broadcast
FP32 inputs. `Footprint.lean` proves exact output-prefix writes.

`Bytes.lean` uses an independent raw-byte observer, a one-cell codec and a
one-event frame. `Checked.fp_equiv` derives the two actual FP16 output bytes
from pre-call inputs. `Checked.frame` preserves every byte of every live
caller backing allocation, including bytes outside logical views and
unaligned allocation tails, on success and all modeled ordinary failure
outcomes. The allocation boundary states actual metadata, capacity and
physical separation from caller allocations; it does not add a separately
proved output-base alignment property.

Four local numerical ports remain: FP32-to-FP16 cast, ordered comparison
against zero, FP32 left multiplication by the source slope literal, and
one fixed K8 result-lane ISA profile. They accept unrestricted FP32 words.
There is no arbitrary-K dot, vector activation, kernel or result oracle.
See `FPValueScope.md` and `numerical-obligations.json`. The Real theorem uses
the Real event interpretation; it does not assert native-to-Real equality.

## Validation and preservation

All 38 task Lean modules have compiled objects; the checked public and
supporting axiom audits pass with the standard base only. Executable checks
passed 43 actual repaired-source CPU reference/frame cases, 12 rejections,
5 ordinary failure prefixes and 1 original logical-view counterexample.
These include ragged/grouped grids, K tails, empty/K0 branches, offset and
broadcast B, read-only aliases, and caller backings with extra byte tails.
Real Torch CUDA FakeTensor controls passed 10 metadata/owner/identity cases
and 1 cross-device rejection. No native GPU numerical run was performed;
the CPU interpreter is not TF32/WGMMA conformance evidence.

The 25 historical Lean modules are byte-identical to their historical
versions, with old public files moved to `Proofs/ConditionalCorrect.lean`
and `Proofs/ConditionalFPEquiv.lean`. Their source is
`matmul_leakyrelu_fp8_conditional.py`; their scope is retained in
`README.conditional.md`. The original benchmark Python and the restored
original `Proofs/Source.lean` dependency are byte-identical to the historical
checkout. Shared integration, ledgers and acceptance remain separate.

Pinned SHA256 values are recorded in `source_pin.json`:

- Original: `5d8796c8b490469195f1e25d4807e5a0a214ea65e4481e1b34075ce4b377fa55`
- Checked: `a5a6166c7e57e82e104805f7060e09a472a62d7be35e04ca7a9296bc5b87631e`
- Conditional: `0aadc6c61e44ea0e3c08fa532887ecd79d8b0b89ab351bf7bb756c8a599ce3bc`
