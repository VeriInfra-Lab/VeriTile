# Native semantics and scalar ABI

The actual source uses FP32 storage for X, W and Y. Its output expression casts
FP32 to FP16 and then back to FP32 before storing. Both casts occur in the shared
source quotation and the independent native expression. All input bit patterns
are admitted; there is no finite-data assumption.

Epsilon is a runtime FP32 scalar, not a constexpr or a numerical literal port.
`EpsilonABI.Binding` relates the actual Python binary64 word, its exact finite
rational value and the launcher's FP32 argument. `convert` uses the core IEEE
rounding definition with round-to-nearest, ties-to-even. The existing finite
positive host guard allows positive underflow to zero and positive overflow to
infinity; Lean fixtures check both and a halfway tie. The source execution is
seeded with that runtime word and its independent native target uses the same
leaf. Compiler shape selectors cannot inspect that word.

`Native.Instructions` interprets individual literals, binary operations, unary
operations, format conversions, integer conversions and FP32 fused multiply-add.
There is no output, tensor, whole reduction or kernel instruction port.
`Native.Reductions` chooses an explicit binary scalar tree, with a proof that all
lanes occur once and any extra neutral leaves are counted. It sees the complete
reduction layout and expression shapes, with numerical data erased.

`Native.Schedule` exposes source-structural compiler choices: split or adjacent
FP32 mul/add contraction, operand order for commutative instructions, elimination
of the literal-zero initial accumulator, same-format FP32 cast elimination,
width-one division elimination, and lowering `1 / sqrt(v)` to the scalar rsqrt
instruction. The last four are explicit flags with source-preserving defaults.
The FP16 output round trip remains explicit. These choices select scalar
instruction graphs; no algebraic floating-point identity is used to equate two
lowerings or to assert reassociation, FMA = split, or sqrt/div = rsqrt.

The native result is exact for every such instruction interpretation and
structural profile: actual typed source and candidate execute successfully, their
observed result is the independent native graph, all actual output Tensor words
and bytes agree, and the entire byte memory equals the independent update.
No numerical equality assumption is required. Binding a chosen scalar profile
to a particular Triton/GPU compiler artifact is an external conformance task;
CPU wrapper controls are not evidence of that binding.

The Real interpretation shares the source quotation but binds epsilon to the
host's mathematical value and erases representation casts. It proves the
conventional RMSNorm formula and denominator positivity under the original
positive-epsilon guard. It is separate from, and does not claim accuracy bounds
for, the rounded FP32/FP16 computation.
