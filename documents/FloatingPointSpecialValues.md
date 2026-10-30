# FP32 exponential special values

`VeriTile/Triton/Float/ExponentialExact.lean` proves `EXP-ZERO` and
`EXP-NEG-INF-SUB` for two distinct FP32 specification profiles. The special-value
table comes from NVIDIA's published specification. The finite nonzero
approximation algorithm remains a parameter; this file does not implement that
algorithm, establish its error bound, or prove a compiler/hardware implementation
conforms to the specification.

## Sources and binding

The original exponential registry uses FP32 inputs, compute, accumulators and
outputs, Triton 3.7.1 and CUDA 13.0. The formal model preserves that scope.

| Primitive | Source and relevant statement |
| --- | --- |
| `ex2.approx.f32` | [PTX ISA, CUDA 13.0.0, §9.7.3.21](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#floating-point-instructions-ex2): `-Inf → +0.0`, `-0.0 → +1.0`, `+0.0 → +1.0`, `+Inf → +Inf`, `NaN → NaN`. |
| CUDA FP32 `expf` | [CUDA Math API, CUDA 13.0.0, single precision](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html): `expf(±0)` returns 1, `expf(-∞)` returns +0, `expf(+∞)` returns +∞, and `expf(NaN)` returns NaN. |
| libdevice ABI | [CUDA 13.0.0 `__nv_expf`](https://docs.nvidia.com/cuda/archive/13.0.0/libdevice-users-guide/__nv_expf.html) gives `float @__nv_expf(float %x)` as the base-e exponential entry. [Triton v3.7.1 CUDA libdevice wrapper](https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/language/extra/cuda/libdevice.py) binds FP32 `libdevice.exp` to this symbol; it is distinct from `__nv_fast_expf` and from FP64 `__nv_exp`. The model selects the documented CUDA `expf` special-value semantics for this ABI. |
| Triton FP32 `tl.exp` | [Triton v3.7.1 `ExpOpConversionApprox`](https://github.com/triton-lang/triton/blob/v3.7.1/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp#L634) emits FP32 multiplication by `log2e = 1.4426950408889634`, then `llvm.nvvm.ex2.approx.f32`. The FP32 constant is `0x3fb8aa3b`. |

The model uses the existing nearest-even, no-flush FP32 scalar multiplication
for that lowering. It does not replace `tl.exp` with Real exponentiation or with
`libdevice.exp`. The source precision and the multiplication before `ex2` remain
part of the definition.

The original `Exponential` entries are all FP32. Supplemental BF16 or mixed
BF16/FP32 records require their own cast and primitive-profile connection;
`ExponentialProbeExact.lean` supplies that connection for the original probe
expressions, as described below. The FTZ variant has the same listed special
values, but its finite/subnormal behavior is a separate profile.

## What is proved

`specialValueModel` implements the complete documented special-value dispatch.
The caller supplies finite nonzero behavior and a NaN-result function whose
result is a NaN. It does **not** supply either target rewrite as an equation.
The two primitive indices keep their residual algorithms distinct. Both input
zero signs are recognized separately; the output one and output zero are the
exact positive FP32 words. NaN sign/payload equality is not promised.

`neg_inf_sub` proves `FP.sub cfg (-∞) x = -∞` for every finite FP32 word and
every scalar rounding/flush configuration. The finite domain includes both
zero signs and all subnormals; NaNs and both infinities are excluded. This proof
uses the existing scalar implementation, its sign-bit negation and decoding.

`tl_exp_zero`, `tl_exp_neg_inf_sub`, `libdevice_exp_zero`, and
`libdevice_exp_neg_inf_sub` then prove the special-value rewrites, including the
actual `tl.exp` multiplication step. All twelve public checked results pass
`#axiomsClean`; no numerical report or empirical admission appears in their
proofs. The fixed backend may choose canonical NaN as a documented software
profile, while the generic theorems leave the payload choice open.

These are exact facts of the selected specification model. Connecting a source
intrinsic to that model is an explicit ABI/profile binding, not a theorem that
all hardware implementations have identical output bits. `EXP-SUB` is not a
consequence of this model and continues to require the project's numerical
admission. Existing experiment artifacts remain historical evidence; no new
experiment was run for these exact special-value facts.

## Original probe profiles and actual fragments

`Exponential.lean` connects the four FP32 special-value atoms to actual
`run`/scalar-output observations in the fixed `Software.strictFP32` backend.
Their `Atom.entry?` is `none`, so their historical reports do not enter `R`.
The public rewrite API still preserves the two intrinsic identities.
`EXP-SUB` is selected through empirical evidence as before.

`ExponentialProbeExact.lean` separately models the original supplemental
Python probe expressions with a small syntax indexed by the backend:

| Rule | Reference | Candidate |
| --- | --- | --- |
| `EXP-ZERO` | `store(rnd(exp(+0)))` | `store(+1)` |
| `EXP-NEG-INF-SUB` | `store(rnd(exp(rnd(-Inf - load(a)))))` | `store(+0)` |

The `Profile`, `load`, `rnd`, and `store` definitions come from
`ScalarProbeExact.lean`. FP32 uses FP32 storage and computation. BF16 uses BF16
storage, loads to FP32 and rounds each marked intermediate through BF16 back
to FP32. The mixed BF16/FP32 profile uses BF16 storage with FP32 intermediates.
Every profile retains the final storage conversion. The exponential primitive
itself always receives FP32; no BF16 exponential instruction is invented.

`Certificate profile backend rule` proves equality of the two fixed source
executions for every allowed residual algorithm. `EXP-ZERO` has no input
condition; `EXP-NEG-INF-SUB` requires the stored input to be finite, including
both zero signs and all subnormals. The proof uses finite-preserving loads,
the exact FP32 subtraction and exponential facts, and exact preservation of
zero, one and negative infinity through the relevant conversions. Separate
theorems identify the final stored zero and one words.

The current catalogue uses eight distinct cases: both tl rules at the three
existing profiles, and both libdevice rules at FP32. The generic composition
theorem does not expand that catalogue or admit new experiments. These eight
cases carry typed execution certificates, not a theorem-name string or the
presence of a historical passing report.

## Reproducibility

Documentation and compiler source were read on 2026-10-21. Downloaded copies are
under `.veritile-cache/strict-fp-laws-20261021/exp-sources/`. SHA-256 digests:

| File | SHA-256 |
| --- | --- |
| `ptx-13.html` | `e91dafce5083999eec0a2d5c86143edd3fb9cdd4b592f031ef8bc236c279dd78` |
| `cuda-math-13-single.html` | `725ecb05f9ad8b0670a7bbfa8a4b21ec173d8afaad6661e06c41d0ffb459beba` |
| `libdevice-13-expf.html` | `10ae4c8423d9b5d1f2dfe84c9c89eb4b3dca66bd91ccbd4a3856e3c081dc3b3b` |
| `triton-3.7.1-lowering.cpp` | `90261ec90c0f47e8d42f9c4aca94f5608cb26d3083bd0e7c05204353c1eed540` |

Build command:

```text
python .veritile-cache/strict-fp-laws-20261021/build.py VeriTile.Triton.Float.ExponentialExact
python .veritile-cache/strict-fp-laws-20261021/build.py VeriTile.Triton.Float.Exponential
python .veritile-cache/strict-fp-laws-20261021/build.py VeriTile.Triton.Float.ExponentialProbeExact
```
