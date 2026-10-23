import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Host

namespace VeriTile.Bench.Repairs.RmsnormFused
open VeriTile Triton
noncomputable section

/-- Even formed addresses of inactive padding lanes fit signed 64-bit byte
address arithmetic. Active loads/stores have the sharper allocation bound. -/
theorem padded_byte_bound (q : Host.Request) (row k i : Nat)
    (hr : row<q.rows) (hk : k<q.block) (hi : i<4) :
    4*(row*q.columns+k)+i<2^63 := by
  have rows := q.gridFits
  change q.rows≤2^31-1 at rows
  have width : q.columns≤16384 := le_trans q.valid.2.1 q.valid.2.2.1
  have block : q.block≤16384 := q.valid.2.2.1
  have prod := Nat.mul_le_mul (show row≤2^31-1 by omega) width
  omega

theorem whole_grid_trace_safe (q : Host.Request) (X Y W : RegionName)
    (hXY : X≠Y) (hWY : W≠Y) (bounds : RegionBounds)
    (hx : q.size≤bounds X) (hy : q.size≤bounds Y) (hw : q.columns≤bounds W)
    (row : Nat) (hr : row<q.rows) (s : BlockState) :
    (kernel X Y W q.columns q.block q.epsilon).eraseDType.toAlgKernel.TraceSafeR .triv bounds
      (Real.setup q.rows row s) := by
  apply VeriTile.Bench.TritonBenchG.RmsnormFused.General.typed_full_trace_safe
    .fp32 .fp32 .fp32 Bool.false bounds (Real.setup q.rows row s) X Y W
    q.columns q.columns q.block q.epsilon (block_positive q.geometry q.valid) hXY hWY
  · intro k
    exact lt_of_lt_of_le (offset_bounded q.geometry row k.val hr k.isLt) hx
  · intro k
    exact lt_of_lt_of_le k.isLt hw
  · intro k
    exact lt_of_lt_of_le (offset_bounded q.geometry row k.val hr k.isLt) hy

#axiomsClean padded_byte_bound
#axiomsClean whole_grid_trace_safe
end
end VeriTile.Bench.Repairs.RmsnormFused
