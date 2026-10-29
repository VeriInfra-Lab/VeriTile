import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardPrimitives

/-! Actual second backward-loop reductions, decay, and pointer subtraction.
Subtraction is kept as the source operation even for terminal unused pointers. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondPrimitives
open VeriTile.Triton Primitives
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem dk_eval (p : Layout) (u : BlockState)
    (h : Fin p.BK → Fin p.BV → ℝ) (v : Fin p.BV → ℝ)
    (hh : u.regs .real [p.BK,p.BV] "d_h" = some (matrix p.BK p.BV h))
    (hv : u.regs .real [p.BV] "_v" = some (vector p.BV v)) :
    evalOp (BackwardSource.dkOp p) u = some (vector p.BK (fun k => ∑ j : Fin p.BV,h k j*v j)) := by
  apply BackwardPrimitives.reduce_rows p.BK p.BV _ u (fun k j => h k j*v j)
  simp only [evalOp.eq_def,hh,hv,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k,j,⟨⟩⟩
  simp [matrix,vector,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex]

theorem dv_eval (p : Layout) (u : BlockState)
    (h : Fin p.BK → Fin p.BV → ℝ) (k : Fin p.BK → ℝ)
    (hh : u.regs .real [p.BK,p.BV] "d_h" = some (matrix p.BK p.BV h))
    (hk : u.regs .real [p.BK] "_k" = some (vector p.BK k)) :
    evalOp (BackwardSource.dvOp p) u = some (vector p.BV (fun j => ∑ i : Fin p.BK,h i j*k i)) := by
  apply BackwardPrimitives.reduce_columns p.BK p.BV _ u (fun i j => h i j*k i)
  simp only [evalOp.eq_def,hh,hk,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [matrix,vector,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex]

theorem decay_eval (p : Layout) (u : BlockState) (h : Fin p.BK → Fin p.BV → ℝ) (b : ℝ)
    (hh : u.regs .real [p.BK,p.BV] "d_h" = some (matrix p.BK p.BV h))
    (hb : u.regs .real [] "b_b" = some (Tile.scalar (some b))) :
    evalOp (.mul .real .scalarR (.ref .real [p.BK,p.BV] "d_h") (.ref .real [] "b_b")) u =
      some (matrix p.BK p.BV (fun k j => h k j*b)) := by
  simp only [evalOp.eq_def,hh,hb,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k,j,⟨⟩⟩
  simp [matrix,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex]

theorem retreat_pointer (n : Nat) (r : RegionName) (ptr : RegName) (addr : Fin n → Nat)
    (stride : Nat) (u : BlockState)
    (hp : u.regs .ptr [n] ptr = some (pointers n r addr)) :
    evalOp (.ptrSub .scalarR (.ref .ptr [n] ptr) (.constNat stride)) u =
      some (pointers n r (fun j => addr j-stride)) := by
  simp only [evalOp.eq_def,hp,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp [pointers,Tile.vec,Tile.ptrSub,Broadcast.leftIndex,Broadcast.rightIndex]

#axiomsClean dk_eval
#axiomsClean dv_eval
#axiomsClean decay_eval
#axiomsClean retreat_pointer
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondPrimitives
