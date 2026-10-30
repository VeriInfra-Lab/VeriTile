# Scope and evidence

The domain is the canonical repair's ordinary exact contiguous CUDA FP32
payload Tensors, zero storage offset, int32 length/table metadata, positive even
head width, valid grouped Q/K heads, independently pitched Cos/Sin rows, both
old and split K-cache layouts, and separate live backing storages. The new
predicate is `tokens = 0 or fullBlock <= 1048576`. Existing shape, address,
metadata, per-axis grid and pointer checks are unchanged. The guard is placed
immediately after calculating the actual full tile. Accepted live calls have
positive power-of-two half and full tiles no larger than Triton's recorded
frontend tensor limit. This is a static frontend bound, not a per-device
resource cap.

The complete quotation includes all source loads, signed int64 conversions,
metadata-dependent offsets, Q low/high stores, the GQA leader branch, K
low/high cache stores, and the actual single full-width V load/store. Exact
signed division/remainder are connected to natural metadata only after the
admitted positive length facts are proved. The grid proof executes the quoted
statement semantics at each real program id and threads final memory; it does
not assume successful execution, an output invariant or a destination callback.

Real correctness is the exact-real projection. Native correctness is over
arbitrary raw FP32 words and explicitly parameterized local `mul`, `neg`, and
`fma` instructions. Its independent graphs are:

- Q low: `fma(q0, cos, neg(mul(q1, sin)))`.
- Q high: `fma(q1, cos, mul(q0, sin))`.
- K low: `fma(cos, k0, neg(mul(sin, k1)))`.
- K high: `fma(sin, k0, mul(cos, k1))`.
- V: copy the initial word without floating arithmetic.

No commutativity of native multiplication is assumed. The direct product
execution retains the Q/K profiles separately and observes each output at its
selected coordinate. The native theorem connects these executed results to
the existing finite uint64 byte execution before applying actual FP32 Tensor
logical observations and frames. The seven FP32 Tensor bindings contain only
metadata, identity, storage and index laws. Int32 metadata retains the existing
checked descriptor and physical-byte interpretation.

Local model boundaries are Torch API binding laws, individual native primitive
semantics, and compiler realization of the selected scalar graph. They are
stated boundaries, not deferred numerical experiments. The eight saved actual
JIT SM80/SM90 profiles are reused because the JIT source and launch arithmetic
are unchanged; all 32 IR/PTX artifacts are hash checked, and their 40 exact
ordered Q/K/V store dataflows are audited. This evidence does not assert a
universal compiler theorem or native GPU execution. No new GPU or offline
compilation is needed for this guard-only candidate. No whole-domain resource,
allocator or machine-launch theorem is claimed. Grid-product compatibility
remains deferred as requested.

The boundary controls capture the actual wrapper launch path for both layouts:
D=1048576 is admitted with half/full tiles 524288/1048576; live D=1048578 is
rejected before metadata copies/launch; empty D=1048578, 2097152, and 67108864
still return without either action. These are host captures, not GPU launches.
The frozen original defect retains the two actual frontend compile failures
at D=1048578 and the D=2 old-layout/D=6 new-layout controls. Historical controls
retain 43 admissions, 27 rejections, 9 complete interpreter grids and 3 original
source controls. Reused evidence is identified separately from new controls.

New candidate Lean modules have current source-matching workspace receipts.
The packet also pins the complete current repository source import closure,
first-resolved objects and their symlink targets, including inherited original
rotary/shared objects. Missing inherited source-build receipts/interfaces are
explicitly recorded. Historical root-review pins establish preservation; they
do not claim that inherited objects were rebuilt from the current sources.
The packet is a candidate delivery for independent review, not acceptance or a
ledger/count update.
