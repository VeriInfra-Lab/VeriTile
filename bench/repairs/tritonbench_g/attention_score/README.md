# attention_score: direct contiguous FP32 contract

This revision repairs the original query-tail and signed-window arithmetic
failures while retaining its column-score API, grouped KV heads, both load
specializations and one persistent block-size retry. Inputs are passed directly;
there are no normalization copies. The original benchmark is untouched (SHA256
`08c87e7a44ce401d01eeadcf2e81b3500bc30e5d31565041026cc33caf08920b`).

## Supported call and independent target

`get_score(q, k, m, sliding_window, complement_sliding_window)` accepts ordinary,
contiguous strided CUDA FP32 tensors on one device, without unresolved negative
or conjugate flags. Q has shape `[Z,H,N,D]`, K `[Z,HK,NK,D]`, M `[Z,H,R]`, with
`R >= N`, positive HK dividing H, and D a power of two at least 16. Read-only
inputs may alias, and contiguous storage offsets are allowed. Both persistent
block globals must agree and be 16, 32 or 64. Window offset/size are signed-int32
integers; the complement flag is Boolean.

The fresh returned FP32 tensor has shape `[Z,H,NK]`. For each output key j,
its independent Real target is

```
sum over i < N with allowed(i,j):
  2 ** ((sum over d < D: Q[z,h,i,d] * K[z,h//(H/HK),j,d])
         * (1/sqrt(D)) * 1.4426950408889634 - M[z,h,i])
```

Without a window all query/key pairs are allowed. The ordinary window is
`0 <= i-j+offset < size`; the source's complement mode is
`i-j+offset >= size`. It is not the full Boolean complement of the first mask.
Empty queries return zeros, and empty outputs return without launching.

Nonempty launch checks bound metadata and dense strides by signed int32, padded
block coordinates and the final query advance by signed int32, byte
displacements by signed int64, the second grid dimension by 65535, and every
formed byte pointer by unsigned int64. Actual input/output pointers must be
4-byte aligned. These checks include masked lanes and final pointer advances;
empty calls do not query or constrain unused pointers.

## Source repair and precision

The query-contribution mask uses `IS_EVEN_M`; it no longer incorrectly follows
key divisibility. The loop ends at N, independent of padded M capacity, and M
loads are masked. Window distances, head offsets and output offsets widen
before arithmetic. Scalar casts use `tl.cast`, which accepts integer values
specialized as constexpr 1 by the actual JIT binder. The typed source's `.to`
node denotes the same signed-int64 cast.

The successful first launch or single OutOfResources retry returns the same
allocated object. A retry halves both block globals with floor 16 and persists
both values. Allocation, selected device context, zero return, and persistent
global state are represented by the public host composition. Failed launches
are outside the normal-return conclusion.

Native arithmetic retains default TF32 QK instructions (`m16n8k8`), FP32
storage, FP32 host-scale conversion and multiplication by rounded log2(e), the
observed scale-first fused multiply/subtract, `ex2.approx.ftz.f32`, ordered
column reductions and FP32 stores. Fusion is explicit. The Real theorem erases
rounding and source casts; it does not assert exact-real GPU arithmetic.

## Proof entries and boundary

- `Kernels.lean` contains the typed sources and imports no task proof.
- `Correct.score_real` proves the complete independent column formula,
  returned object/shape/device/dtype, both persistent globals, machine integer
  closure and the physical exterior frame.
- `FPEquiv.score_native` independently executes the typed source into exact
  native instruction graphs from the current FP32 input words. It includes the
  same host result, globals, machine closure and physical frame.
- `FPEquiv.score_native_if_local` transports that result between instruction
  interpretations using only scalar operations and a coordinate of a fixed
  `m16n8k8` MMA. No reduction, whole matrix product, output-memory equality or
  source result is a numerical premise.
- `Proofs/Contract.lean` states classified metadata, shape/window and launch
  checks. `check_iff` identifies its typed Boolean admission predicate with the
  named contract; `domain` and `aligned` derive the finite integer/address
  premises. `Shape`, `Dispatch` and ordinary storage/Tensor bindings separately
  record dimensions, successful retry dispatch and actual contiguous readback.
  The retained helper name `Normalized` now binds direct logical inputs; it
  performs and assumes no copy.

The trusted execution boundary is ordinary Tensor metadata/contiguity,
successful fresh allocation, Python/typed-source correspondence and successful
race-free runtime/compiler realization of the recorded local instruction
routing. Offline compilation provides evidence for the selected profiles, not
a verified Python/Triton compiler. The native host scale is the actual positive
host scalar supplied to the launch; IEEE instruction conformance remains a
local numerical obligation. Historical ideal-native-to-Real helpers are
retained under `Proofs/`; they are not the current public numerical claim.
The shared large-grid launcher issue remains separately deferred in
`bench/TODO.md`.

## Validation

All 59 Lean modules compile under Lean 4.29.0; public roots and their helpers
pass standard-base axiom and noncircular-target audits. There is no new axiom
or `sorry`.

- `python bench/repairs/tritonbench_g/attention_score/check_contract.py`:
  62 admissions, 19 rejections, 32 complete CPU interpreter grids with input
  frames, and three original query-tail/window-overflow/lazy-negative controls.
- `check_cuda_lowering.py`: 11 actual-JIT-binder SM80 profiles, covering every
  block size, both even-load branches, both window modes, tails and scalar
  N/offset/size=1 specialization. Artifacts default to
  `/tmp/veritile-attention-score-lowering`.
- `check_precision.py` checks current source casts and precision against the
  hash-bound IR/PTX artifacts.

CUDA execution was unavailable. CPU interpreter and offline compiler checks do
not validate GPU numerical conformance.
