# Batched vecmat complete-program schedule proof

This isolated package contains three new scheduling modules and the unchanged
independently reviewed `HostReadback` helper: four Lean modules, one public
specification. It does not change repository sources or promote whole-program
acceptance.

`batched_vecmat_arbitrary_program_order_correctness` runs the complete original
fp32 logical source for any permutation of the complete floor-divided host grid.
Each program starts with fresh registers and reads the preceding actual memory.
The final memory equals both an independent reference that reads only initial
inputs and the previously proved `gridExpectedMemory`; the full frame is kept.

The explicit profile condition is `A ≠ output ∧ B ≠ output`. A and B may alias
each other. Distinct programs' output addresses are proved disjoint from the
actual host grid and lane layout. No successful-run, output-equality or
arithmetic/reassociation premise is introduced. One chosen `FP.Config` and one
family of valid `ReductionPlan`s remain fixed throughout; the theorem does not
equate different trees or claim that they match a native compiler.

The proof first identifies the actual body's A/B load-register values and exact
element offsets, proves input-region preservation through complete execution,
and derives disjoint program write sets. The readback helper turns the existing
scatter reference into an exact value at each writing lane. Actual current-memory
program effects then equal fixed initial-input effects. Disjoint writes give
commutation, and list permutation gives the complete schedule theorem.

The internal partial-schedule theorem permits any list of valid program IDs,
including empty lists and repetitions. It does not admit extra out-of-grid IDs:
this source uses unmasked stores. The public complete-grid contract requires a
permutation, so each valid ID occurs once. Empty grids and K=0 are included
without additional positive-dimension premises. For nondivisible sizes the
parameter-level grid uses the same floor division as `gridCount`; original
wrapper guards are covered by the existing separate host theorem.

This closes only arbitrary sequential order of complete logical programs under
the explicit separated profile. Physical tensor/storage offsets, byte aliases,
allocator freshness, fixed-width arithmetic, native compile/launch admission,
and instruction/warp interleavings remain structural obligations. Native FP
primitive/reduction-tree conformance remains in the existing numerical follow-up;
no new numerical law or deferral is added.

The final receipt binds every new source, the recursive project import graph,
original Python and wrapper sources, actual compiler exits, before/after source
hashes, compiler logs, and exact public statement hash. Dependency object symlinks
and temporary `.olean` outputs are compilation aids, not integration sources.
`HostReadback` retains its separate producer receipt and root review; its already
successful unchanged compile was not repeated.

Suggested integration adds the stage `separated_fp32_arbitrary_sequential_program_order`,
preserving prior declarations and the shared DSL source view's
`single_program_total_typed_memory` label. Root must review and bind the new stage
metadata separately. A standalone integration build and current catalog/global
gates are still required after source integration.
