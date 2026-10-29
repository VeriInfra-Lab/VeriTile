import bench.tritonbench_g.matmul_persistent_triton.Proofs.NumericWitness
import bench.tritonbench_g.matmul_persistent_triton.Proofs.ZeroKWitness
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealSource
import bench.tritonbench_g.matmul_persistent_triton.Proofs.IterationSchedule

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
#print axioms FPProof.source_equivalence
#print axioms SourceChecks.persistent_structure
#print axioms Schedule.ordinal_bound_iff
#print axioms Schedule.tile_covered
#print axioms Schedule.tile_unique
#print axioms Schedule.grouped_bounds
#print axioms Schedule.ki_next
#print axioms RealMath.partial_complete
#print axioms RealSource.projection
#print axioms NumericExecution.nonempty_typed_execution
#print axioms ZeroK.run_unchanged
#print axioms ZeroK.zero_K_counterexample
#print_fp_assumptions FPProof.source_equivalence
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
