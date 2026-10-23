# Native numerical boundary

Each source iteration forms matching FP16 or BF16 products, reduces 256 values
in that same format, casts the sum to FP32, and adds it to an FP32 accumulator.
Scaling occurs once after the loop. The result is converted to the FP32 output
format and enters its own atomic addition. K=0 still performs scaling on the
initial FP32 zero and issues all 64 atomic contributions.

Installed Triton 3.7.1 selects no floating promotion for either FP16 or BF16 in
`_pick_sum_dtype`. Its reduction semantic returns the input scalar dtype. The
sum/selector sources and version are pinned in `numerical_profile.json` and
checked by the executable checker. The historical numerical source is retained.

`Direct.ScalarTree.Plans` gives each reduction a finite typed scalar tree with
every original lane once and its declared padding-zero leaves. Each join calls
one scalar addition. Shape, axis, keepDims, operand order, zero padding, product
format, extension, cycle accumulation, scaling and output conversion remain
explicit. No whole reduction callback appears in the native acceptance result.

`NativeGraph.Instructions.atomicAdd` is a separate two-operand instruction at
the storage dtype. It is not ordinary FP32 addition. This preserves a boundary
for native atomic rounding and denormal behavior. The reference folds the 64
payloads in each cell's actual runtime order. No FP associativity, commutativity,
Real projection, atomic reorder, or numerical whole-result equation is assumed.

The local runtime trace records individual atomic transitions. Exact event
coverage and source-to-transition correspondence are proved. The compiler
binding records the whole production AST, actual selected trees and structural
primitive lowering. Numerical ports are fixed-arity local instructions only.
Raw input words and concrete output bytes come from the original caller windows.

The Real result separately proves the full K-term sum using the partition of K
among splits/cycles/lanes. It does not justify floating-point reassociation.
Source/CPU checks do not instantiate GPU instruction behavior on all special
values or claim native GPU numerical conformance.
