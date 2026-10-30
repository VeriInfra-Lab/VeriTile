import bench.repairs.tritonbench_g.kldiv_ops.Proofs.NativeBinding

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage
noncomputable section

/-- The signed row, row-stride product, padded full element offset, loop
start, block, and lane all fit i64. The only size bound comes from allocation. -/
theorem dense_intermediates (t : Tensor) (dense : Dense t) (rows cols row start lane : Nat)
    (shape : t.numel=rows*cols) (hr : row<rows) (hs : start<cols) (hl : lane<blockSize cols) :
    (BitVec.ofNat 64 row).toInt=(row : Int) ∧
    (BitVec.ofNat 64 cols).toInt=(cols : Int) ∧
    (BitVec.ofNat 64 (row*cols)).toInt=(row*cols : Nat) ∧
    (BitVec.ofNat 64 (row*cols+start+lane)).toInt=(row*cols+start+lane : Nat) ∧
    (BitVec.ofNat 64 start).toInt=(start : Int) ∧
    (BitVec.ofNat 64 (blockSize cols)).toInt=(blockSize cols : Int) ∧
    (BitVec.ofNat 64 lane).toInt=(lane : Int) := by
  have allocated := dense_count_bound t dense
  have hc : 0<cols := by omega
  have hm := flat_index rows cols row start hr hs
  have rowBound := Nat.mul_le_mul_left row (show 1≤cols by omega)
  have colsBound := Nat.mul_le_mul_right cols (show 1≤rows by omega)
  have capped := block_capped cols
  simp only [Nat.mul_one,Nat.one_mul] at rowBound colsBound
  rw [shape] at allocated
  norm_num at allocated ⊢
  exact ⟨Native.signed_exact _ (by omega),Native.signed_exact _ (by omega),
    Native.signed_exact _ (by omega),Native.signed_exact _ (by omega),
    Native.signed_exact _ (by omega),Native.signed_exact _ (by omega),Native.signed_exact _ (by omega)⟩

/-- An empty-width grid has no lane arithmetic or loads. For reduced output,
the actual row allocation still bounds the row and its one-word output. -/
theorem ForwardPlan.reduced_row_integer {rows cols : Nat} {prediction target output : Tensor} {red : Reduction}
    (plan : ForwardPlan rows cols prediction target output red) (reduced : red≠.none)
    (row : Nat) (hr : row<rows) : (BitVec.ofNat 64 row).toInt=(row : Int) := by
  have allocated := dense_count_bound output plan.outputDense
  have count : output.numel=rows := by simp [Tensor.numel,plan.outputShape,Reduction.outputShape,reduced]
  apply Native.signed_exact
  rw [count] at allocated
  norm_num at allocated ⊢
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host
