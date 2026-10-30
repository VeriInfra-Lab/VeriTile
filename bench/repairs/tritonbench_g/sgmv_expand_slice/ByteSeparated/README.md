# sgmv_expand_slice

Ordered segment fold: each active output is sum_{k<inner} X[row,k]*W[adapter,column-slice,k], optionally plus previous output; inactive/untouched cells retain previous value.

- [sgmv_expand_slice_byte_separated.py](../sgmv_expand_slice_byte_separated.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete original ordered source jobs, full K loops, all eight signed metadata ABIs, both ADD modes, skipped/empty/tail/singleton paths and overlapping-segment order. Readback covers all output coordinates; writable footprint only active cells.

## Conditions

- Input/weight/output ABI remain single/half/single.
- PrivateRegions; every segment fits; inner/columns/slice equal logical Shape fields; slice+columns≤outputColumns; contiguous output layout.
- PublicIO.metadataDomain binds only live segments signed start/length/adapter values; skipped metadata remains unconstrained.
- Contract.fitted/capacity derive the corresponding logical fit/capacity conditions from accepted metadata.
- The public IO judgment quantifies over logical inputs, physical memory binding and capacity bounds; source safety, successful execution, output values and exact exterior frame are conclusions, not caller premises.

Exact FP program pairs and numerical dependencies are recorded in the [sgmv_expand_slice contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
