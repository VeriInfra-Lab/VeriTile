# Numerical statement

The original numerical graph is retained. FP32 input is converted to FP16,
multiplied by FP16 weights, and reduced by an explicit selected FP16 scalar
tree. The sum widens to FP32 before the optional FP32 add and FP32 store.
Padded products and zeros are explicit. The selected output conversion is a
contraction boundary.

`Native.Schedule` records compiler choices based only on erased graph shape:
operand swaps and left/right adjacent FP16 mul/add contraction. Six local
scalar primitive families implement narrow, widen, mul16, add16, fma16 and
add32. There is no whole-reduction or whole-output numerical callback.
All raw words, including NaN payloads, signed zeros and subnormals, remain
visible to those local instruction ports. Positive-zero conversion identities
are explicit local compiler obligations because the retained symbolic source
normalizes constant-zero casts.

`Contract.power_profile` kernel-checks the installed six-shift power helper
for every admitted K. `source_pin.json` pins original/candidate/quotation and
the unchanged kernel AST; `numerical_profile.json` pins installed sum, sum dtype
selection and power helper sources. Installed Triton retains the input FP16
sum dtype. No accidental FP32 reduction replaces it.

The compiler binding pins the production AST, exact reduction plans and
shape-based selection. It assumes equality only for individual scalar
instruction families. Full source execution, output expressions, raw store
readback and byte frames are proved. Actual compiler/native instruction
conformance remains that local external boundary; no GPU execution claim is
made. CPU AST checks exercise control, indexing and source arithmetic shape.

The independent Real target uses an ordinary finite K-term sum plus optional
old Y, preserving sentinel rows. Its proof evaluates the symbolic source with
identity casts and exact arithmetic; it does not equate floating results with
real arithmetic.
