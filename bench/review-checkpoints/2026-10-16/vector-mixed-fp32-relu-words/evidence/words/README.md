# ReLU finite input words and typed-grid refinement

This isolated extension adds three modules and one public specification. It
reduces part of RELU-S4 by making all nine integer/bool input representations
explicit: signed and unsigned BitVec 8/16/32/64 words, and a raw BitVec 8 bool
byte. It does not change the six frozen ReLU modules or establish whole-original
program acceptance.

IntegerWords proves source-word decoding bounds and signed/unsigned word
roundtrips. Small signed inputs sign-extend to i32 for comparison; small
unsigned inputs zero-extend and compare signed at i32. Larger signed inputs
compare signed and larger unsigned inputs compare unsigned at their source
width. Signed comparison agrees with nonnegativity of the decoded two's
complement value; every unsigned comparison succeeds. All 256 bool bytes are
normalized by !=0 to unsigned i1 before comparison and conversion. The fp32
conversion uses the original source-width value, preserving the observed
route rather than converting the widened comparison operand.

The finite-word comparison/select/conversion formula is proved equal to the
existing typed result on a decoded input cell. InputWords requires equality
of the complete input cell, including its tag, at every offset below N. It
does not silently accept wrong-tag Cell.read defaults. wordSpec uses the
initial words and arbitrary prelaunch memory, with no interpreter result in
its definition.

The public relu_word_grid_correctness composes this relation with actual
successful execution of the existing full ascending typed grid. Only pid zero
writes; each active first-block result is tagged fp32 and every other address
retains prelaunch memory. No successful kernel run is assumed by this theorem.
The actual-result helper identifies any returned state with that already
proved result and supplies exact readback plus the full frame.

No logical input/output separation is required for the relation theorem.
withInputWords also constructs a concrete logical state from finite words;
the separate y != x lemma preserves a separately supplied output region.
If prelaunch output cells are already fp32-tagged, the untouched tail retains
that tag. Neither statement proves fresh allocation or physical non-overlap.
Shifted or mixed-width byte aliases, endian assembly, alignment, storage
bounds, native loads, and compiler correspondence remain structural work.

M.fromInt, M.fromNat and the fp32 zero literal remain explicit callbacks.
No numerical identity, real exactness, finite-value condition or native
rounding behavior is assumed. The public cmp16/cmp32 parameters are inherited
from the existing backend but unused by the nine integer/bool routes. fp16,
bf16, fp32, fp64, fp8 errors, other dtype branches, native finite-width address
arithmetic, native host execution and instruction scheduling are not extended
by this batch.

evidence/route-review.json binds all thirteen successful TTIRs from the prior
seventeen-profile source audit and annotates the nine routes used here. These
are the existing pinned sm90 frontend observations; no backend compilation or
GPU run is newly claimed. The Lean word definitions and proofs do not replace
a source-to-native compiler or actual memory-decoding proof.

The frozen receipt records only final actual Lean exits, log and source
hashes, axiom/noncircularity checks, and before/after source dependencies. All
six prior ReLU sources, the actual lakefile, manifest, toolchain and original
Python source are bound. External library builds are not rechecked here.
