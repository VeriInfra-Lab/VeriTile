import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalLoop
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Scope
namespace VeriTile.Bench.Repairs.LayerNormOps.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section

def atPid (s : State ℝ) (pid : Nat) : State ℝ :=
  {s with pids := fun ax => if ax=0 then pid else s.pids ax}

/-- Summing the actual source row-loop accumulators covers every input row
once, including M=0 and idle programs. The initial +0 and every row addition
were retained in the source proof; only this ideal-real interpretation uses
associativity of real addition. -/
theorem host_dw (rest : Algebra ℝ) (fp : ForwardParameters) (bp : BackwardParameters)
    (rows programs : Nat) (h : Scope.Supported fp bp rows programs) (s : State ℝ)
    (hf : InputsFresh bp) (i : TileIndex [bp.block]) (hi : i.1.val<bp.columns) :
    (∑ pid ∈ Finset.range programs, (Backward.accumulated (algebra rest) bp (atPid s pid)).dw i) =
      ∑ row ∈ Finset.range rows, rowDW bp s row i.1.val := by
  simp_rw [loop_dw rest bp _ hf i hi]
  simp only [Backward.start,Backward.stop,atPid,if_true,rowDW]
  exact Scope.partition_sum fp bp rows programs h (fun row => rowDW bp s row i.1.val)

theorem host_db (rest : Algebra ℝ) (fp : ForwardParameters) (bp : BackwardParameters)
    (rows programs : Nat) (h : Scope.Supported fp bp rows programs) (s : State ℝ)
    (hf : InputsFresh bp) (i : TileIndex [bp.block]) (hi : i.1.val<bp.columns) (hb : bp.hasBias=Bool.true) :
    (∑ pid ∈ Finset.range programs, (Backward.accumulated (algebra rest) bp (atPid s pid)).db i) =
      ∑ row ∈ Finset.range rows, rowDB bp s row i.1.val := by
  simp_rw [loop_db rest bp _ hf i hi hb]
  simp only [Backward.start,Backward.stop,atPid,if_true,rowDB]
  exact Scope.partition_sum fp bp rows programs h (fun row => rowDB bp s row i.1.val)

/-- Every active contiguous FP32 matrix address spans a complete word inside
its allocation. These bounds do not assume arbitrary layouts or offsets. -/
theorem complete_word_bound (rows columns row col : Nat) (hr : row<rows) (hc : col<columns) :
    4*(row*columns+col)+3 < 4*(rows*columns) := by
  have hmul : (row+1)*columns≤rows*columns := Nat.mul_le_mul_right _ (by omega)
  nlinarith

/-- Independent positive strides for saved statistic views are read exactly
at their logical row addresses, within the minimal backing span. -/
theorem statistic_word_bound (rows stride row : Nat) (hr : row<rows) :
    4*(row*stride)+3 < 4*((rows-1)*stride+1) := by
  have hx : row≤rows-1 := by omega
  have hm := Nat.mul_le_mul_right stride hx
  omega

#axiomsClean host_dw
#axiomsClean host_db
#axiomsClean complete_word_bound
#axiomsClean statistic_word_bound
end
end VeriTile.Bench.Repairs.LayerNormOps.Logical
