# Finite-precision rotary contracts

`FPCorrect.lean` contains complete per-program value and memory-frame contracts
for fp32 Q/K and fp32 precomputed cos/sin buffers. Both branches execute all four
original stores. Sizes, row/head strides, program id, tails and initial memory
are symbolic; aliasing follows the original ordered writes. `referenceMemory`
reads only the initial input snapshot and specifies explicit addresses and
separate products followed by addition/subtraction and storage casts.

`rope_software_fp32` gives an independent `roundedMemory` formula using actual
binary32 `FP.mul`, `FP.add` and `FP.sub`. No callback from the surrounding backend
occurs in that formula. Each operation rounds separately with `FP.Config`.
`rope_nonempty_rounding` checks all four outputs of a nonempty tile, with masked
head/column lanes and an unchanged neighboring cell. Its exact value
`1 - 2^-25` rounds to `1`, so the witness distinguishes the model from real
arithmetic. The scalar bit calculation is checked by the Lean kernel.

`Proofs/TypedShape.lean` proves the erased typed source equals the full existing
source, including both branches, all loads, addresses, masks and stores.
`Proofs/TypedSource.lean` preserves load-to-register conversions and materializes
implicit destination-storage conversions. Erasure is a structural bridge; the
new execution proof retains the conversions and arithmetic precision.

For hardware use the intended profile must have fp32 storage and computation,
separate multiply and add/subtract (`enable_fp_fusion=false`), and a rounding,
DAZ/FTZ and NaN policy matching the selected config. `Config.ieeeValue` uses
nearest-even without DAZ/FTZ; ScalarOps canonicalizes arithmetic NaNs. Exact NaN
payload correspondence has not been established. Native pointer widths,
allocation bounds, race freedom and host grid/contiguity also need their usual
bindings. These software theorems do not certify the default fused compiler
specialization. Other storage-format combinations remain outside this fp32
value contract, even though the typed source definition retains both format
parameters. Cos/sin producers remain outside the previously accepted scope.
