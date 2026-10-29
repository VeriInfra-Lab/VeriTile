import bench.tritonbench_g.uniform_sampling.Correct
import bench.tritonbench_g.uniform_sampling.FPEquiv
import bench.tritonbench_g.uniform_sampling.Proofs.WordArithmetic
import VeriTile.Meta.StatementAudit

/-! Public contracts and complete-execution dependencies must use only the
standard logical base. The public floating relation has no numerical atoms.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling

#axiomsClean Formula.expected_at
#axiomsClean Formula.four_fields
#axiomsClean SourceDomain.active_coverage_unique
#axiomsClean SourceDomain.counter_index_lt_i32
#axiomsClean CounterOperands.wrapped_low_add
#axiomsClean RandomDomain.all_inputs_valid
#axiomsClean RandomDomain.field_value
#axiomsClean TypedRandom.eval_philox
#axiomsClean TypedRandom.eval_uniform
#axiomsClean Erasure.projection_ok
#axiomsClean Erasure.complete_projection
#axiomsClean Erasure.full_statement_count
#axiomsClean Algorithm.source_eq
#axiomsClean Execution.source_exec
#axiomsClean RealIO.trace_safe
#axiomsClean RealIO.source_correct
#axiomsClean RealSurface.realizes_frame
#axiomsClean Correctness.uniform_sampling_correctness
#axiomsClean Correctness.index_commuted_correctness
#axiomsClean Correctness.nonempty_instances
#axiomsClean FPProof.whole_state
#axiomsClean TypedExecution.source_executes
#axiomsClean TypedExecution.nonempty_pair
#axiomsClean TypedExecution.all_four_streams_active
#axiomsClean TypedExecution.tail_masked
#axiomsClean TypedExecution.counter_wrap
#axiomsClean Witness.real_nonempty
#axiomsClean Equivalence.uniform_sampling_equiv
#axiomsClean Equivalence.nonempty_pair
#print_fp_assumptions Equivalence.uniform_sampling_equiv

end VeriTile.Bench.TritonBenchG.UniformSampling
