import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardVariantIO
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardFP
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedCandidate

/-!
Complete forward control paths with explicit proof scopes.

Forward.ksoftmax_forward_fp32_variant_real_correctness gives both actual
fp32 original and candidate finite flat IO after mathematical dtype erasure.
It covers all LOG, CAUSAL and IS_FP16 flags, None/qk/bk masks, every 2D program
ID, arbitrary source strides and positive tile depth. M is read only when
present and read-active; output stores use k < K, including causal future
lanes. The mathematical specification uses scalar finite supremum and sum.

Forward.Floating.forward_index_equiv establishes typed whole-state/failure
equality with no floating assumptions for four declared storage triples.
It retains arbitrary Option String values, including unknown strings, and
commutes only natural multiplication in the X row address. All floating
operations, casts, controls and failure paths remain in the source.

Forward.TypedExecution.fp16_log_causal_qk_both_values and its bk companion
connect the actual original/candidate typed runs to distinct nonzero mask
inputs. Written future lanes retain bottom in these typed cell statements;
the erased real-memory formula observes bottom with unbotD 0. Neither fact
is a native IEEE execution guarantee.

These contracts replace the retired plain-only forward entry. The complete
backward contracts and their mathematical source are independent.
The 216-triple source AST audit is separate from typed execution coverage;
partial fp64 mask-pointer joins remain rejected by the current frontend.
The host launch, heuristic flags, finite machine address arithmetic, native
compiler admission and inter-program interactions remain external to these
per-program formal contracts.
-/

#axiomsClean VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.ksoftmax_forward_fp32_variant_real_correctness
#axiomsClean VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Floating.forward_index_equiv
#axiomsClean VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Floating.forward_run_eq
#axiomsClean VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedExecution.fp16_log_causal_qk_both_values
#axiomsClean VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedExecution.fp16_log_causal_bk_both_values
