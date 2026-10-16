import bench.tritonbench_g.seeded_dropout.Correct
import bench.tritonbench_g.seeded_dropout.FPEquiv
import bench.tritonbench_g.seeded_dropout.Proofs.TypedExecution
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.seeded_dropout.Proofs.RealIO
import bench.tritonbench_g.seeded_dropout.Proofs.RandomDomain
import bench.tritonbench_g.seeded_dropout.Proofs.FlatMemory
import bench.tritonbench_g.seeded_dropout.Proofs.Memory
import bench.tritonbench_g.seeded_dropout.Proofs.Index
import bench.tritonbench_g.seeded_dropout.Proofs.Formula
import bench.tritonbench_g.seeded_dropout.Proofs.SourceDomain

namespace VeriTile.Bench.TritonBenchG.SeededDropout.SourceDomain

#print axioms legalBlock_arange_checks
#print axioms wrapper_block_legal
#print axioms grid_zero
#print axioms grid_le_i32_blocks
#print axioms launched_block_start_lt
#print axioms launched_lane_lt_i32
#print axioms source_intermediates_i32
#print axioms source_operations_exact
#print axioms source_mask_exact

end VeriTile.Bench.TritonBenchG.SeededDropout.SourceDomain

namespace VeriTile.Bench.TritonBenchG.SeededDropout
#print axioms Index.offset_injective
#print axioms Index.offset_div
#print axioms Index.offset_mod
#print axioms Index.offset_pair_unique
#print axioms Index.distinct_programs
#print axioms Index.quotient_lt_grid
#print axioms Index.active_coverage_unique
#print axioms Index.active_offset_lt_i32
#print axioms Formula.expected_eq_if
#print axioms Formula.expected_congr_input
#print axioms Formula.expected_keep
#print axioms Formula.expected_drop
#print axioms Formula.expected_zero_input
#print axioms Formula.expected_of_one_le
#print axioms Formula.expected_of_neg
#print axioms Formula.canonical_decoder
end VeriTile.Bench.TritonBenchG.SeededDropout

namespace VeriTile.Bench.TritonBenchG.SeededDropout.Memory
#print axioms writeSequence_cell
#print axioms writeSequence_state
#print axioms mem_addresses
#print axioms storeBlock_scatter
#print axioms storeBlock_cell
#print axioms storeBlock_active_cell
#print axioms storeBlock_frame
#print axioms storeBlock_state
end VeriTile.Bench.TritonBenchG.SeededDropout.Memory

namespace VeriTile.Bench.TritonBenchG.SeededDropout.FlatMemory
#print axioms bounds_of_extents
#print axioms storeBlock_active_cell
#print axioms storeBlock_frame
#print axioms storeBlock_readback
end VeriTile.Bench.TritonBenchG.SeededDropout.FlatMemory

namespace VeriTile.Bench.TritonBenchG.SeededDropout.RandomDomain
#print axioms arange_domain
#print axioms all_inputs_valid
#print axioms eval_rand
end VeriTile.Bench.TritonBenchG.SeededDropout.RandomDomain

namespace VeriTile.Bench.TritonBenchG.SeededDropout
#print axioms Erasure.kernel
#print axioms Algorithm.source_eq
#print axioms Execution.prefix_exec
#print axioms Execution.exec_prefix
#print axioms Execution.prefix_rand
#print axioms Execution.executes
#print axioms Execution.exec_raw
#print axioms Execution.output_active
#print axioms Execution.raw_eq_storeBlock
#print axioms Execution.exec_storeBlock
#print axioms Execution.resultBase_mem
#print axioms Safety.prefix_safe
#print axioms Safety.suffix_safe
#print axioms Safety.trace_safe
#print axioms Safety.flatten_ok
#print axioms Candidate.prefix_flat
#print axioms Candidate.algorithm_flat_exec
#print axioms Candidate.flat_exec
#print axioms RealIO.resultBase_flat_mem
#print axioms RealIO.original_correct
#print axioms RealIO.candidate_correct
#print axioms RealSurface.outside_iff
#print axioms RealSurface.projection
#print axioms RealSurface.prefix_run
#print axioms RealSurface.algorithm_exec
#print axioms RealSurface.executes
#print axioms RealSurface.realizes_frame
end VeriTile.Bench.TritonBenchG.SeededDropout

namespace VeriTile.Bench.TritonBenchG.SeededDropout
#print axioms Correctness.seeded_dropout_correctness
#print axioms Correctness.index_commuted_correctness
#print axioms FPProof.prefix_run
#print axioms FPProof.suffix
#print axioms FPProof.whole_state
#print axioms TypedExecution.one_legal
#print axioms TypedExecution.prefix_executes
#print axioms TypedExecution.prefix_seed
#print axioms TypedExecution.prefix_offsets
#print axioms TypedExecution.prefix_rand
#print axioms TypedExecution.suffix_executes
#print axioms TypedExecution.original_executes
#print axioms TypedExecution.candidate_executes
#print axioms TypedExecution.nonempty_pair
#print axioms Equivalence.seeded_dropout_equiv
#print_fp_assumptions Equivalence.seeded_dropout_equiv
end VeriTile.Bench.TritonBenchG.SeededDropout
