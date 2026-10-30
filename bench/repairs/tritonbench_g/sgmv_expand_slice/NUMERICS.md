# Numerical statement

The accepted kernel graph is retained: each K16 block uses
`tl.dot(a, b, accumulator)`, with FP16 operands and an incoming FP32
accumulator. A 32x32x16 source step consists of eight disjoint native
m16n8k16 instruction ports. It is not replaced by scalar FMA or an independent
dot followed by addition. Source K-block ordering and incoming accumulators
are explicit.

The source casts FP32 X to FP16. After accumulation it casts FP32 to FP16.
The add branch promotes the computed half value to FP32, combines it with
old Y using the accepted old-first native operand order, and stores FP32.
Replacement promotes the rounded half value to FP32. K padding preserves
its explicit zero conversions. Two local zero-conversion laws are the
retained native adapter boundary. NaN payload/sign/rounding behavior is
provided only by exact local conversion, add and m16n8k16 ports.

`Native.Primitives` has no reduction, tile-result or complete-output callback.
`Native.result` independently forms fragment operands, executes the ordered
K blocks, applies the cast/add epilogue and folds segments in batch order.
`NativeBackend.prepared_result` proves complete source execution. The direct
binding instantiates its pure source state from original raw bytes and checked
metadata. Its historical region names do not require physical private clones.

`Proofs/Direct/Compiler` pins the whole production AST and requires only exact
source compilation and local primitive agreements. Actual compiler/hardware
instruction conformance remains a local external boundary; CPU AST tests do
not certify it. No GPU execution claim is made. `source_pin.json` binds the
original, candidate, quotation and unchanged historical kernel AST.

The Real theorem has a separate exact-real interpretation of the same typed
source. Identity floating casts yield ordinary finite K-term contractions and
serial segment composition. It does not claim the floating program equals
real arithmetic.
