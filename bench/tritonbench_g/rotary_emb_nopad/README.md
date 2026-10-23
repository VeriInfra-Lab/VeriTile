# rotary_emb_nopad

- Source file: `rotary_emb_nopad.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

## Concrete fp32 coverage of all four selected faces

`FPCorrect.lean` exposes separate fp32 correctness specifications for block Q,
block K, V2 Q, and V2 cache. These are the four value sub-kernels already
selected by `rotary_emb_nopad_output_summary_general`; none is replaced by a
smaller half-store fragment. The full-source lowering facts remain available
in the legacy entry. Python tests explicitly use fp32 Q, K, cosine, sine and
cache (`rotary_emb_nopad.py:323–337`). The new sources use `DTYPE = .fp32`,
retaining fp32 memory, register, load and store types. These Python paths use
binary subtraction, not unary negation, so no sign-flip rewrite is introduced.

Block Q/K retain the complete `[BLOCK_TOKENS, 1, HEAD_HALF]` tile, token/head
masks and the K group-leader guard. V2 Q/cache retain their head masks,
token-masked coefficient loads, and the cache's full sequence-length/block-table
chain. All four contracts derive both output halves from the initial-memory
inputs, run concrete `FP.mul` followed by `FP.sub` / `FP.add` with separate
rounding at each operation, prove successful execution and both readbacks,
and preserve every cell outside the two active write sets. No abstract
arithmetic callback occurs in the resulting scalar formulas.

`Pair.WritesApart` requires injectivity within each active output half and
disjointness between active halves. These are the existing address-side
conditions from the legacy contracts, expressed per initial state and only
for active lanes. No finite-value assumption, exact-rounding law, assumed
output, new token-validity gate or new stage reduction is introduced. The
in-place block/Q sources load both halves before either store.

V2's explicitly specified cosine/sine defaults are positive zero. The local
`FPV2.zeroDefaults` supplies only the concrete real-literal +0 and its conversion
to binary32 +0; it does not assume an identity conversion on arbitrary reals.
`FPV2.software_correctness` proves that the result depends only on this covered
zero case and concrete fp32 arithmetic.

There is a necessary correction to the old mathematical V2 cache address
reference: the masked block-table load has **no `other` argument**. On an
out-of-range token its value is `initial.undef .nat`, not an asserted zero.
The new independent address formula preserves that unspecified state value
and proves execution/readback/frame for every choice. For valid tokens it is
the ordinary table value. This keeps the source's full mask domain rather
than introducing a token-validity assumption. The four source-projection
results match the full existing selected programs after dtype erasure; they
do not assert that the old hardcoded-zero reference describes an unspecified
native load.

Nonempty witnesses start with empty registers. Block Q and K each rotate one
active row with a masked token tail; V2 Q and cache each execute both stores,
with cache outputs reached through nonzero block/slot metadata. All four
witnesses compute `1 ± 2^-25` as bits `0x3f800000`, retain a neighboring cell,
and include a changed input/output half. A further V2 cache witness uses an
out-of-range token, explicit coefficient zeros, and the nonzero unspecified
block value 7; it proves stores at 704/705, not fictitious block-zero addresses.

Full fused-stage/host composition remains outside the accepted faces.
CUDA/compiler fusion, FTZ/NaN policy and machine-address refinement remain
separate native bindings; the proved numerical profile is concrete software
fp32 with per-operation rounding. The legacy natural-number metadata/address
model is retained.

Validation logs are `/tmp/rotary-nopad-{pair,blocks-execution,blocks-contract,
v2-execution,v2-contract,software,witness,public}.log`; helper and public
modules contain `#axiomsClean` checks.

A separate [direct guarded repair](../../repairs/tritonbench_g/rotary_emb_nopad/README.md) provides complete arbitrary-Real and raw-FP32 wrapper proofs, including the paged-cache branch. The original Python remains unchanged.
