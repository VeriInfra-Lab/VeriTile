# softmax_flaggems finite seed, v2 native replica binding

This is a local native-semantics revision of the frozen `softmax_flaggems_finite_seed`
candidate. The Python file is byte-identical to v1: only the two online maxima
change from `-inf` to finite FP32 minimum `0xff7fffff` relative to the selected
source. All guards, admitted input words, casts, tiles, branches and fusion
remain unchanged. The selected source and canonical selection are untouched.

## F1 and the repaired interpretation

The v1 independent review is preserved with decision `CHANGES_REQUIRED`.
Its F1 finding showed that one scalar reduction expression per logical row
cannot represent the saved B32/C32 inner-backward lowering. That PTX performs
`FMA(local P, local G, rounded butterfly partner product)` separately in each
physical lane, then consumes that lane's reduction replica directly. With
`P=1/32`, `G[0]=17*2^-149`, `G[16]=-16*2^-149`, and other G zero, the retained
analytic derivation gives scale `2^-149` versus `+0` and stored gradients `+0`
versus `-0`. No common FP32 row scale matches both stores.

V2 represents a logical scalar as a function from its actual destination
consumer `(region, flat store address)` to a scalar instruction expression.
Every reduction receives both the logical reduced coordinate and that consumer;
its selected scalar tree can therefore differ by physical lane/replica.
Fusion and multiplication operand selection also receive the consumer. This
interpretation applies to all four kernels, one-tile/online paths and all six
backward configurations. It never assumes replica equality or restricts
subnormal/exceptional input words.

The returned Tensor contains evaluated raw words, not producer expressions.
Each subsequent kernel starts with `Native.entry` leaves containing those fixed
words. `Native.stored_word_handoff` and `FPEquiv.tensor_word_handoff` prove that
changing a later consumer or profile cannot reinterpret a producer expression.
The invalid v1 native autograd endpoints are not v2 claims; v2 adds no new
host/autograd project.

## Source, mathematical target and Tensor boundary

`Kernels.lean` contains complete ordinary FP32 quotations for the four selected
forward/backward source functions and their four existing address candidates.
`Proofs/ReadableSource.lean` proves all eight complete function equalities to
the prior definitions by `rfl`, for every argument and both tile branches.
Their complete branch/loop/grid proofs are reused with the new algebra.
`Correct.lean` reuses the reviewed v1 independent WithBot/Real softmax and VJP
proofs unchanged: finite or negative-infinity logits, at least one finite entry
per active column, and finite actual saved-output/upstream values for the VJP.
NaN, +inf and all--inf remain admitted; they have no claimed normalized Real
result. This mathematical scope is not a runtime guard.

The v2 public native endpoints are `FPEquiv.checked_forward` and
`FPEquiv.checked_backward`. They connect complete quotation execution to the
independent consumer-indexed native graph, actual allocator Tensor metadata,
raw output words, four-byte readback and exterior frame. The input binding
uses the same physical `before` memory as the returned `post`: input words are
assembled from the actual Tensor `dataPtr` bytes. Backward has a separate actual
saved-output descriptor; the original x is only its shape/device reference.

Ordinary observed full backing extents and destination freshness prove that
every byte of each input's entire storage, including a view's prefix/suffix,
is preserved. The theorem also derives unchanged input-word readback from post.
Saved output and upstream storage may alias each other. These are allocation,
metadata and byte-binding observations, not extra Python guards or a proof of
PyTorch's allocator implementation. Empty outputs require no separation and
perform no stores.

## Retained compiler profile mapping

`profile-mapping.json` pins all 36 existing backward PTX files: six inner and
six noninner configuration families, each with saved N1, N3 and N1025 tags.
Choice `j` has tile `32*2^j`; the other tile dimension is `max(1,1024/tile)`.
N1 is a constexpr specialization. Other tags compile N as runtime i64 and fix
`ONE_TILE_PER_CTA` according to the tag. Their host-consistent scope is recorded
per artifact: `1<=N<=TILE_N` for one-tile, `N>TILE_N` for online. In particular,
noninner N3 with TILE_N 1 or 2 is online. These are manual-signature SM80
artifacts, not evidence of the exact Python JIT binder specialization.

For every profile the v2 public theorem uses the same consumer-indexed R/S
interface. The actual compiler supplies local scalar trees, fusion/operand
selection, and any selected shared publisher as value-independent structural
ports. A shared load binds its actual published operand; no numerical equality
between all possible producers is required. The mapping retains shuffle,
shared load/store, FMA, subtraction-consumer and global-store instructions for
review. The two unchanged v1 changed-forward artifacts are retained too; their
final shared broadcast can select the same published tree for every consumer.

`Replica32` concretely instantiates the saved one-tile B32/C32 inner profile.
Lean proves lane coverage for all 32 consumers, source-scale binding, the
ordered distance16 FMA, later distances8/4/2/1, the destination-address-to-lane
mapping, and the final source sub/multiply/store operand order. Its total-algebra
fallback is not claimed to model another compiler profile. Other configurations
remain on the explicit compiler-selected R/S boundary; none uses the obsolete
single-row carrier or silently inherits Replica32's fallback tree.

## Validation and freeze

Ten v2 modules have current successful source/object receipts. No v1 or shared
module was rebuilt. The read-only profile checker validates eight PTX row-register
consumer chains and rejects five instruction/profile mutations. It verifies the
unchanged v1 source/quotation identities and reuses the accepted four full AST
checks, eight source mutants, 22 regression cases, 21 rejections, five historical
counterexamples and 40 boundary cases. The reviewer analytic counterexample is
retained rather than re-executed. There are no new compilations, numerical
experiments or GPU runs.

Evidence is in `.veritile-cache/softmax-flaggems-finite-seed-v2-evidence/`.
`delivery.json` identifies the superseded v1 packet and preserved review, all
v2 sources/receipts/effective imports, the 36 saved backward artifacts and the
reused source/Real/current-forward compiler evidence. `verify_delivery.py` is
read-only. The compiler, fixed-arity instruction behavior, selected structural
profiles and ordinary Tensor/backing observations remain explicit trust
boundaries. No universal compiler/resource success or native equality to the
unrounded Real ratio is claimed.
