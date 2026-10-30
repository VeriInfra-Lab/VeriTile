# Source and model scope

The Python source is unchanged. It has one `@triton.jit` entry,
`rmsnorm_triton`, and the wrapper chooses `out = torch.empty_like(x)`,
`BLOCK_N_SIZE = 4096`, `num_warps = 16`, and a two-dimensional `(batch, M)`
grid. No source dtype or signed-stride guard restricts the input and weight.

`Kernels.lean` models both complete `range(0, N_SIZE, BLOCK_N_SIZE)` loops,
both program IDs, all seven strides, every masked load and each in-loop store.
Its 216 floating carrier profiles independently select input, weight and output
from fp16, bf16, fp32, fp64, fp8e4nv and fp8e5. The wrapper's floating subset has
output storage equal to input storage. The candidate only commutes the natural
multiplication in the first term of `offset_m`.

Pass one converts each loaded input to fp32 before squaring. The accumulation,
sum, division by N and square root are fp32. Pass two reloads the original input
without that cast, divides by the fp32 standard deviation, then multiplies by
the original weight. Input fp64 promotes the division; weight fp64 can promote
the later multiplication independently. The typed macro spells out the ordinary
pointer store's implicit destination conversion. Unlike a block-pointer store,
the source `tl.store` admits that conversion. `Proofs/Precision.lean` independently
prescribes both loop bodies' loads, arithmetic scopes and destination conversion.

Both input loads have `other=0.`. The masked weight load omits `other`; the typed
source retains an undefined masked value and does not invent zero padding.
Structural FP equivalence is for an arbitrary backend and arbitrary undefined
values, including failures and aliases. The exact-real projection instead
requires an explicit zero-undefined convention in its successful IO predicate.
It uses mathematical real arithmetic and identity storage casts. Its actual
trace-safety predicate does not assume undefined values or successful execution.

`full_real_correct` quantifies symbolic N, B, epsilon, both program IDs through
the state, all seven nonnegative strides, all floating profiles and both variants.
Its explicit preconditions are B positive, output column stride positive,
output distinct from both inputs, finite disjoint allocations, all active access
bounds, and (for real execution) zero undefined values. The independent formula
sums all active input lanes across every tile. Successful flat execution and the
complete raw-cell frame are in the named `FullExactIO` surface. Historical
`RoundingModel` contracts are preserved and retain their erased-real meaning.

## Offline native compiler evidence

`Proofs/native_lowering.py` extracts the unchanged JIT AST without importing or
running the module-level CUDA tests. The recorded Triton 3.7.1 / CUDA sm90 probe
uses N=4097, B=4096, eps=1e-6, 16 warps, three stages, and signed i32 runtime
strides. Thus both loops have two iterations and a partial final tile.

- 32 of 36 floating wrapper profiles compile to PTX; their TTIR has the expected
  fp64 division/multiplication positions and fp32 square root.
- Four wrapper profiles fail during LLVM lowering: fp64 input with either fp8
  weight requires an unsupported fp8-to-fp64 conversion, and either fp8 input
  with fp64 weight requires an unsupported fp64-to-fp8 destination conversion.
  The formal carrier family does not supply native admission for those cases.
- Four additional floating output combinations compile: `(fp64, fp16, bf16)`,
  `(fp16, fp64, bf16)`, `(fp64, fp64, fp32)`, and `(fp32, fp32, fp64)`.
- Four nonfloating probes also compile: `(i32, fp32, i32)`, `(fp32, i32, fp32)`,
  `(i1, fp32, i1)` and `(i32, i32, i32)`. They demonstrate source paths outside
  the current floating formalization; they are not proofs of those paths.

The receipt `Proofs/native_lowering_sm90_triton_3_7_1.json` records source and
compiler-semantic SHA256 hashes, exact profiles, compilation outcomes and the
conversion diagnostics. Expected native rejection is not counted as successful
native execution or as a verified numerical profile.

## Remaining source obligations

The whole source port is not complete. Integer/Boolean and other storage labels,
signed strides with shifted bases, fixed-width index overflow and native shape
admission beyond the recorded target remain outside the current formal family.
The real theorem's symbolic positive B does not assert native admission of every
B; native `tl.arange` shape restrictions still apply. Implicit conversions,
floating special values and rounding are represented by an abstract backend;
no IEEE bit-accuracy or GPU numerical result is claimed. Host allocation and
launch, cross-program races, scheduling and independent comparator replay remain
separate unverified obligations. The official comparator environment is unavailable.
