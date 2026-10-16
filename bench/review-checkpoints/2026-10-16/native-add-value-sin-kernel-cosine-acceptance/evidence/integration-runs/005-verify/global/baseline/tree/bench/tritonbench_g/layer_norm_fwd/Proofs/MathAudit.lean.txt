import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layer_norm_fwd.Proofs.Domain

/-! Audit only the mathematical checkpoint. No source execution, correctness,
finite IO or FP equivalence is certified by this file. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

#axiomsClean Formula.seedU32_range
#axiomsClean Formula.seedU32_of_host_range
#axiomsClean Formula.dropout_zero
#axiomsClean Formula.dropout_keep
#axiomsClean Formula.dropout_drop
#axiomsClean Formula.stream_padded
#axiomsClean Formula.combined_padded
#axiomsClean Formula.sum_padded
#axiomsClean Formula.sum_combined_padded
#axiomsClean Formula.variance_nonneg
#axiomsClean Formula.rstd_pos
#axiomsClean Contract.expectedCell_none_iff
#axiomsClean Contract.post_frame
#axiomsClean Contract.inRow_of_col
#axiomsClean Contract.inRow_iff_col
#axiomsClean Contract.rows_disjoint
#axiomsClean Domain.eval_seed_cast
#axiomsClean Domain.col_i32
#axiomsClean Domain.random_inputs_valid
#axiomsClean Domain.eval_rand
#axiomsClean Domain.row_address_lt
#axiomsClean Domain.row_address_lt_span
#axiomsClean Domain.second_row_lt
#axiomsClean Domain.first_mask_lt
#axiomsClean Domain.second_mask_lt
#axiomsClean Domain.mask_streams_disjoint

end VeriTile.Bench.TritonBenchG.LayerNormFwd
