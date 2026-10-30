# Kernel revision: blocked on shared minimum support

This is a preparatory assessment, not acceptance of the new kernel revision.
The authoritative `contract_revision` selects
`chunked_cumsum_fwd_repaired.py`, SHA256
`5f834ef4712a5f138681eb9c97ec332c43119e716e4d3c82fa8537fc0ec02d2a`.
The Python, guards, `Kernels.lean`, `Correct.lean`, and `FPEquiv.lean` are
unchanged. Historical contract acceptance is retained in its recorded scope.
No minimum staging build or integration was performed.

For an output coordinate `(b,h,k,j)`, let `t = k * chunk_size + j`.
The independent Real formula adds the optional head bias to `dt[b,t,h]`,
optionally applies `log(1 + exp(x))` for `x <= 20` and `x` otherwise,
then applies the lower maximum followed by the upper minimum. The prepared
output is this value when `t < seqlen`, and zero otherwise. The cumulative
output is the inclusive sum of these prepared values times `A[h]`, from
lane zero through lane `j` within the same chunk. Both outputs have shape
`(B,H,ceil(S/chunk_size),chunk_size)`; the function returns cumulative first.

For native FP32, preparation retains casts, configured addition operand
order, both separate runtime clamp words, the cutoff comparison, and final
remasking. The prefix retains its selected ordered multiply/add/FMA tree.
There is no assumption of FP associativity, multiplication distribution, or
FMA equivalence to a rounded product followed by addition. In particular,
zero padded prepared values times an infinite or NaN rate may produce NaN;
the Real padding identity must not be applied to arbitrary native words.

The admitted domain includes noncontiguous and zero-stride FP32 inputs,
input aliases, all four bias/softplus flag combinations, all seven head
blocks, empty axes, and Python `True` as an integer chunk size. It imposes
no finite-value, clamp-order, or finite-rate restriction. The finite/default
positive-infinity Real projection is separate from native arbitrary-word
semantics. No new source defect was found in this assessment.

Existing proofs to reuse after the blocker is released:

| Obligation | Existing source | Scope to preserve |
| --- | --- | --- |
| Independent formulas | historical `Proofs/WholeFormula.lean` | Logical initial-array preparation and inclusive prefix |
| Ordered FP graph | historical `Proofs/RawObserved.lean`, `RawObservedIO.lean` | Arbitrary words, separate minimum, structural prefix tree with no payload argument |
| Exact Real projection | historical `Proofs/RawObservedReal.lean`; repair `Proofs/Real.lean` | Tree leaves give the independent prefix; default infinite upper bound handled separately |
| Grid and typed-cell frame | historical `RawBackendGrid.lean`, `RawBackendMemory.lean` | Existing spliced-source execution only, not a new full DSL theorem |
| Tensor/index connections | repair `Proofs/Contract.lean`, `Native.lean`, `Storage.lean` | Original strides/pointers, uint64 masks, same-descriptor four-byte bounds |
| Runtime clamp conversion | historical `Proofs/RawHostABI.lean` | Exact binary64-to-FP32 RN-even algorithm, signed zero, overflow, subnormals, quiet NaN payloads |

The full-source blocker is concrete. `RawSource.Code` separates
`beforeClamp` and `afterClamp`. `RawSource.exec` executes the first list,
replaces the `dt` register using an external `minimum` callback, and executes
the second list. It does not execute one complete authoritative runtime-bound
`ComputeKernel`. The existing shared DSL sends binary `tl.minimum` through
`expandMinMax` to an ordered compare/where; `FP.Structural.Binary` has no
minimum constructor. The public source template also specializes Real
bound literals and uses natural index arithmetic, while the repair has
runtime FP32 bounds and widened uint64 arithmetic. Existing address lemmas
are reusable but do not by themselves prove successful complete finite
source execution. Existing public readers are wrapper outcome theorems
with host/allocator premises.

The shared minimum patch remains frozen. Once authorized and validated by
its owner, the remaining work is to bind both bounds as runtime arguments
in one readable full source, connect the complete finite/native execution,
reuse the formula/grid/frame facts, and expose kernel-only `Correct` and
`FPEquiv` entries. This assessment adds no host, allocator, or unified final
byte-heap obligations, and does not narrow the Python domain or replace
minimum with a comparison workaround.

Two reproducible checks are included:

* `check_kernel_boundaries.py` executes the actual Python AST on CPU backing
  storage with 18 fixtures, each separately checked under all seven head
  blocks: 126 launches and 221 programs. It compares both complete outputs
  to an independent logical-coordinate scalar reference, checks each input
  backing byte and output guards, original input object identity, allocation
  and return order, and unchanged binary64 runtime-bound arguments. Cases
  cover NaN/Inf bounds and inputs, reversed bounds, rate-induced padded NaNs,
  eager overflowing softplus, rounding ties, underflow, aliases, zero strides,
  bool chunks, and all empty-axis kinds. A separate integer RN-even oracle
  matches 4,116 local C double-to-float casts, including signed zeros and NaN
  payload fixtures.
* `check_kernel_lowering.py` compiles the unchanged source offline with
  Triton 3.7.1 for SM80: all seven head blocks and four flag combinations at
  chunk block width four, plus widths one and 64 with both flags enabled
  (42 profiles). Both bounds remain separate FP32 runtime parameters;
  masks use unsigned 64-bit comparisons, and memory accesses are FP32.
  These profiles use integer ABI `i64`. TTIR, TTGIR, LLVM IR, PTX, and the
  local `extractFP32` driver source are pinned in the evidence report.

The offline compiler establishes an important special-value detail:
Triton 3.7.1 defaults to `PropagateNan.NONE`, producing `arith.maxnumf` /
`arith.minnumf` and PTX `max.f32` / `min.f32`. The new boundary replay uses
`torch.fmax` / `torch.fmin` as CPU proxies. The older contract checker uses
propagating `torch.maximum` / `torch.minimum` and only checked finite values
with default bounds; those results are retained without extending their
scope. With a finite left operand and NaN upper bound, the legacy ordered
compare/where selects NaN while the observed minimum returns the finite
operand. Thus the minimum gap affects the admitted domain.

These checks do not establish GPU execution, NaN payload or signed-zero
conformance of GPU operations, every runtime integer signature, a complete
compiler scan-tree binding, or a new Lean kernel execution theorem. The ten
local numerical families remain deferred as recorded in the contract.
