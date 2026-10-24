# ReLU mathematical byte-memory refinement

Four modules connect the nine finite-word ReLU profiles to a genuine byte
memory, `Nat → BitVec 8`. `loadLE` assembles contiguous bytes by concatenation
with the lowest-address byte in the lowest bits. `splitByte` extracts eight
bits; byte stores update those individual bytes. Generic load/store and
decode/re-encode roundtrips cover input widths 8/16/32/64 and fp32 output.

The byte program models the source memory effect: pid zero loads its lane
values from pre-store memory and writes complete four-byte results in lane
order; other programs have no store. Its complete ascending grid equals an
independent byte reference. Exactly the first `min N 1024` output elements
change. Every other byte, including arbitrary output tail bytes, is retained.
Loads in this mathematical memory are total; inactive-lane values and native
access errors are not certified by this memory-effect projection.
The loaded/result functions on the nonzero-pid path can remain unevaluated;
successful native loads on that path do not follow from the byte effect.

`LogicalView` relates actual byte loads to whole tagged input and fp32 output
cells. Under distinct logical names and explicitly separated input/output byte
intervals, the final byte memory represents the actual successful existing
typed grid result. A constructor supplies the logical view for arbitrary
initial bytes without changing those bytes. Neither this constructor nor the
separation premise proves fresh allocation or introduces a source runtime guard.

The public theorem uses the existing fixed-1024, `N < 2^31` domain and explicit
64-bit base/end bounds. It includes byte reads at `BitVec 64` addresses, exact
input-word loads, and every output byte including the last byte. Generic
`loadAt64`/`storeAt64` lemmas connect the mathematical byte operations to the
previously proved finite-pointer arithmetic.

The output carrier is `BitVec 32`; integer-to-fp32 and zero-literal callbacks
remain explicit. Little-endian mathematical memory is a chosen representation,
not a proof of native storage, endianness, provenance, alignment, allocation,
compiler lowering, device dispatch, error behavior or concurrent execution.
The nine dtype profiles are unchanged. Native zero-grid behavior and inputs
outside the bounded domain remain open. Whole-program completion increment: 0.

Every compilation attempt retains its actual exit, log and source snapshot.
Final verification additionally binds all imported project sources/oleans,
the three frozen predecessor receipts, actual lakefile.toml, manifest,
toolchain and original Python source before and after the complete four-module
run. Sources and evidence require independent review before integration.
