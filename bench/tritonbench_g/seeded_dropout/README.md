# Seeded dropout: preserved original source and proofs

The original Python is unchanged. The whole-host historical public statements
and their task-local helpers have been restored alongside the existing
per-program and index-commuted proofs. Their domain and numerical deferrals
remain conditional; the old wrapper itself only asserts contiguity.

The current checked FP32 revision is in
[`bench/repairs/tritonbench_g/seeded_dropout`](../../repairs/tritonbench_g/seeded_dropout/README.md).
It checks its selected domain before allocation and preserves the kernel and
original allocation/assert/launch/return body. It also replaces the stronger
historical padded-pointer-end premise with active allocation bounds and exact
modular inactive-pointer arithmetic.

The original generic `Kernels.lean` keeps its historical source-pin interface,
including the arange-domain proof adapter. The checked revision has a separate
source-only `Kernels.lean`, with a Lean identity bridge to the original source.

## Preserved historical scope

# seeded_dropout

The original pinned wrapper and complete kernel are covered by the additive
`seeded_dropout_real_source_outcomes` and
`seeded_dropout_fp32_source_outcomes` specifications. The original Real and
source/candidate equivalence declarations remain unchanged, including their
wider formal storage profiles. The new whole-task route selects FP32 input and
output because this source does not explicitly choose a production dtype.

The independent Real formula at each original logical flat index `i` is
`x[i] / (1-p)` when the canonical ten-round Philox FP32 random value for
`(seed,i)` is strictly greater than `p`, and zero otherwise. No distribution,
expected-value, or statistical independence assumption is used. The FP32
formula retains load interpretation, canonical RNG bits, strict comparison,
subtraction, eager division, selection and the destination cast. The default
FP32 engine resolves the comparison's actual implicit precision tag; it does
not erase explicit precision or storage operations.

The whole public domain uses `0 <= p < 1`, FP32 contiguous input,
`numel < 2^31`, bounded aligned allocations, and the actual Python integer
seed range `[-2^63,2^64)`. The seed's lossless logical carrier is i32, i64 or
u64 according to `PythonSeedABI`; the installed JIT may further specialize
`seed=1` and `n_elements=1` as constants without changing their values.
The old generic u32 kernel specialization and its supplemental offline compile
are retained, but no ordinary Python-int u32 branch is claimed.

`p` in the public mathematical domain denotes the original finite Python
float value. Its runtime FP32 conversion is separate: a Python value below
one can round to FP32 one. Python `p=-0.0` and `p=+0.0` also have the same Real
value but different runtime bits. Numerical registration therefore uses exact
local conformance laws for the threshold comparison and `1-p`, with the actual
narrowed parameter; it does not assume literal-p bit identity. Literal zero
and one have independent constant rules. No nonzero-denominator, finite-input,
non-NaN, IEEE-RNE-division, or complete-result equality is assumed.

All shapes are admitted, including scalars and zero-size axes. The original
`empty_like` allocation precedes the contiguity assertion. Allocation failure,
assertion failure after allocation, ordinary launch failure and a successful
returned tensor are distinguished. Successful execution uses the entire fixed
1024-wide grid, including inactive tails, in any complete block ordering.
Output shape, dtype, device and contiguous layout match the source.

`WholeTile` reduces every original typed source statement and proves successful
per-block values and full frame. `WholeGrid` proves complete coverage and unique
active output addresses. `WholeHost` composes ordinary allocation metadata with
the actual kernel execution and proves preservation of every preexisting
allocation and every output-exterior cell. `WholeIndex` proves all padded
signed-i32 intermediates, multiplication/addition/masks, exact uint64 byte
pointers, complete four-byte active accesses, alignment, full allocation byte
ends and separation from preexisting allocation bytes. `WholePublic` connects
the independent formulas to the original wrapper outcomes.

The companion checkpoint retains 13 original-AST CPU interpreter value probes
and an allocation-before-assertion probe. The value probes use an independent
Python Philox recurrence and NumPy FP32 reference, including the strict
threshold tie and a probability that rounds to one. Offline Triton 3.7.1/sm90
compilation covers the three dynamic Python-int ABIs, a supplemental u32 ABI,
and the actual seed-one specialization. Its FP32 graph retains ordered `ogt`,
subtraction, division, select and store; PTX uses `div.full.f32`. The exact
installed FP32 parameter extraction helper was also run on CPU. No GPU was run.

Ordinary Torch allocation contracts, value-preserving JIT specialization,
compiler/runtime/load-store correspondence and legal launch scheduling remain
trusted boundaries. Exact numerical primitive conformance is explicitly
pending; offline compilation and CPU examples do not discharge it. Every new
Lean module has a checked standard-axiom audit, and both formulas have checked
noncircularity audits. Producer receipts pin source bytes, final objects,
dependencies, logs and old-public-declaration preservation.
