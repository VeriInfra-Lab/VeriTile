import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardCandidate
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullFP
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardExecution
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedValues

/-!
# Complete k-softmax backward: public proof entry

`Backward.ksoftmax_backward_full_variant_real_correctness` specifies both
actual typed source variants after explicit dtype erasure. Its two inputs and output have
independently chosen storage types and independently chosen row strides.
The theorem includes successful flat execution, the independent backward
formula, active-address safety and preservation of every unwritten cell.
All `LOG`, `CAUSAL` and `IS_FP16` branches remain in the source. In particular,
the logarithmic causal branch writes the negative active-gradient sum at
future lanes that are within `K`.

`Backward.Floating.backward_full_index_equiv` compares the complete typed
original and candidate for every abstract floating algebra, with no numerical
or domain assumptions. The candidate commutes the natural multiplication in
the gradient input address. Floating operation order, reductions, casts,
memory effects and failure remain observable in the equality.

`Backward.RealExecution.all_flags_successful` checks concrete real executions.
`Backward.TypedExecution.source_run` constructs 96 complete nonempty typed
runs in an exact-real algebra: six homogeneous storage profiles, eight flag
combinations and two variants. The two `*_causal_cells` theorems additionally
check actual fp16 interpreter outputs, including the masked tail. These
witnesses do not assert mixed-storage execution under a native IEEE backend.

The source quotation is `backwardSource%` in `BackwardSource.lean`.
`scripts/check_ksoftmax_backward_source_profiles.py` separately compares all
216 original storage quotations against the complete pinned Python AST.
That source check, native compiler profile acceptance and independent proof
replay are separate evidence. In particular the real erasure theorem is not
a numerical error bound, and abstract typed equivalence does not admit every
formal storage profile on native Triton.

The model uses natural indices. The Python host launch, its `DEPTH` and
`IS_FP16` heuristics, machine integer overflow, target instructions and
inter-program races remain outside these per-program contracts.
-/

#axiomsClean VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.ksoftmax_backward_full_typed_real_correctness
#axiomsClean VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.ksoftmax_backward_full_variant_real_correctness
#axiomsClean VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Floating.backward_full_index_equiv
#axiomsClean VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution.source_run
#axiomsClean VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution.log_causal_cells
#axiomsClean VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution.ordinary_causal_cells
