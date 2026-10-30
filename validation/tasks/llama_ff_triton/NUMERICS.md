# Preserved numerical boundary

The FP reference is an independent complete operation graph. K is padded only
through explicit masked zero inputs. Every 64-lane dot block is covered by four
fixed-K16 MMA instructions (`orderedK16`), and the moment reduction uses an
explicit complete `butterfly64` tree. The selected dot/reduction coverage and
source-to-reference graph equalities are proved. No variable-length dot or
reduction callback is a numerical premise.

Remaining numerical interpretation ports are fixed-arity scalar instructions
and one hardware K16 MMA lane component: FP32 literals, add/multiply/divide,
negation, reciprocal square root, sigmoid lowering, FMA, FTZ multiplication,
half product, half widen/narrow, and local E5M2-byte conversions. The libdevice
power lowering retains its two FTZ multiplications and FMA operands/order.
RMS E5M2 conversion goes directly to half; W1/W3 conversion goes through single
then half. X/RMS multiplication retains half rounding before each MMA; both
projection normalizations, SiLU, final product and output half conversion remain
in their accepted order. No cast identities, associativity, FMA-vs-split identity
or FP-to-Real identity are asserted.

`LocalEventABI.stepLaw` describes a single scalar/fixed-K16 event, one current
memory scalar load/conversion, or one scalar store. `CompilerBinding` pins the
complete checked production AST, selected expression/event program, logical
schedule and physical-lane mapping; it contains no output-value equation. The
selected lowering is a structural compiler condition, not a claim that CPU
execution or all compiler versions have established native hardware conformance.
`RawByteABI` gives local byte decoding, and the checked role-width binding fixes
X/output to two bytes and all weight roles to one or two bytes as selected by
the source dtype branch. Mixed-width X/weight backing aliases are rejected by an
explicit production guard; same-width read aliases are permitted.

The Real theorem is separate: storage rounding is erased, a local mathematical
E5M2 decoder is explicit, finite sums are reduced to the independent logical
RMSNorm–SwiGLU formula, and no exceptional byte pattern is silently equated to a
native real number. Historical numerical acceptance and the original numerical
instruction families are preserved.
