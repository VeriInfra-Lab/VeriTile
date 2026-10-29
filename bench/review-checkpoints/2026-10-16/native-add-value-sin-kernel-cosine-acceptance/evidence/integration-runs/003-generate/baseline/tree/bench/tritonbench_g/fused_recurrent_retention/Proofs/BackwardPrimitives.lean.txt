import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSource
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Primitives

/-! Mathematical evaluation of the actual transposed backward state and
reductions. These internal register facts will be established by execution. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardPrimitives
open VeriTile.Triton Primitives
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem state_update (p : Layout) (u : BlockState) (b : ℝ)
    (h : Fin p.BK → Fin p.BV → ℝ) (k : Fin p.BK → ℝ) (v : Fin p.BV → ℝ)
    (hb : u.regs .real [] "b_b" = some (Tile.scalar (some b)))
    (hh : u.regs .real [p.BK,p.BV] "h" = some (matrix p.BK p.BV h))
    (hk : u.regs .real [p.BK] "_k" = some (vector p.BK k))
    (hv : u.regs .real [p.BV] "_v" = some (vector p.BV v)) :
    evalOp (BackwardSource.stateUpdate p) u =
      some (matrix p.BK p.BV (fun k' v' => b*h k' v' + k k'*v v')) := by
  simp only [BackwardSource.stateUpdate,evalOp.eq_def,hb,hh,hk,hv,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k',v',⟨⟩⟩
  simp [matrix,vector,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex]

theorem dq_product (p : Layout) (u : BlockState)
    (h : Fin p.BK → Fin p.BV → ℝ) (d : Fin p.BV → ℝ)
    (hh : u.regs .real [p.BK,p.BV] "h" = some (matrix p.BK p.BV h))
    (hd : u.regs .real [p.BV] "_do" = some (vector p.BV d)) :
    evalOp (BackwardSource.dqProduct p) u =
      some (matrix p.BK p.BV (fun k j => h k j*d j)) := by
  simp only [BackwardSource.dqProduct,evalOp.eq_def,hh,hd,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k,j,⟨⟩⟩
  simp [matrix,vector,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex]

theorem reduce_rows (m n : Nat) (op : Op .real [m,n]) (u : BlockState)
    (f : Fin m → Fin n → ℝ) (hf : evalOp op u = some (matrix m n f)) :
    evalOp (.reduceSum ⟨1,by simp⟩ Bool.false op) u =
      some (vector m (fun j => ∑ k : Fin n, f j k)) := by
  rw [evalOp_reduceSum,hf]
  simp only [Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp only [Tile.reduceSum_false,Tile.reduceSumDrop_data,matrix,vector,
    Tile.mat,Tile.vec,TileShape.axisDim,TileShape.insertAxisIndex]
  change (∑ k : Fin n, ((f j k : ℝ) : WithBot ℝ)) = some (∑ k : Fin n, f j k)
  rw [←WithBot.coe_sum]
  rfl

theorem reduce_columns (m n : Nat) (op : Op .real [m,n]) (u : BlockState)
    (f : Fin m → Fin n → ℝ) (hf : evalOp op u = some (matrix m n f)) :
    evalOp (.reduceSum ⟨0,by simp⟩ Bool.false op) u =
      some (vector n (fun j => ∑ k : Fin m, f k j)) := by
  rw [evalOp_reduceSum,hf]
  simp only [Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp only [Tile.reduceSum_false,Tile.reduceSumDrop_data,matrix,vector,
    Tile.mat,Tile.vec,TileShape.axisDim,TileShape.insertAxisIndex]
  change (∑ k : Fin m, ((f k j : ℝ) : WithBot ℝ)) = some (∑ k : Fin m, f k j)
  rw [←WithBot.coe_sum]
  rfl

theorem dstate_update (p : Layout) (u : BlockState)
    (h : Fin p.BK → Fin p.BV → ℝ) (q : Fin p.BK → ℝ) (d : Fin p.BV → ℝ)
    (hh : u.regs .real [p.BK,p.BV] "d_h" = some (matrix p.BK p.BV h))
    (hq : u.regs .real [p.BK] "_q" = some (vector p.BK q))
    (hd : u.regs .real [p.BV] "_do" = some (vector p.BV d)) :
    evalOp (BackwardSource.dStateUpdate p) u =
      some (matrix p.BK p.BV (fun k j => h k j+q k*d j)) := by
  simp only [BackwardSource.dStateUpdate,evalOp.eq_def,hh,hq,hd,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k,j,⟨⟩⟩
  simp [matrix,vector,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex]

#axiomsClean state_update
#axiomsClean dq_product
#axiomsClean reduce_rows
#axiomsClean reduce_columns
#axiomsClean dstate_update
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardPrimitives
