import bench.tritonbench_g.reversed_cumsum.Proofs.RealLoop
import VeriTile.Triton.Semantics.ProgramRewrite

/-! Exact initialization of both full sources and the initial descending invariant. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealInit
open VeriTile.Triton RealProgram RealStep RealLoop
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def launch (u : BlockState) : Launch := ⟨u.pids 0, u.pids 1⟩

def initial (p : Layout) (u : BlockState) : BlockState :=
  let u1 := u.setReg "i_s" .nat [] (Tile.scalar (u.pids 0))
  let u2 := u1.setReg "i_bh" .nat [] (Tile.scalar (u.pids 1))
  let u3 := u2.setReg "o_i" .nat [p.BT] ⟨fun i => i.1.val⟩
  let u4 := u3.setReg "m_s" .real [p.BT, p.BT] (RealMath.upperTriTile p.BT)
  u4.setReg "b_z" .real [p.BS] (Tile.vec fun _ => some 0)

theorem triangular_eval (p : Layout) (swapped : Bool) (u : BlockState)
    (h : u.regs .nat [p.BT] "o_i" = some ⟨fun i => i.1.val⟩) :
    evalOp (.where (maskOp p swapped)
      (.broadcast (.const 1.0) [p.BT, p.BT]) (.broadcast (.const 0.0) [p.BT, p.BT])) u =
        some (RealMath.upperTriTile p.BT) := by
  cases swapped
  all_goals simp only [maskOp, Bool.false_eq_true, ↓reduceIte, evalOp, evalOp_ref,
    h, Option.bind_some, Option.bind_eq_bind]
  all_goals erw [evalOp_expandDim, evalOp_ref, h, evalOp_expandDim, evalOp_ref, h]
  all_goals simp only [Option.bind_some, Option.bind_eq_bind, Option.pure_def]
  all_goals apply congrArg some
  all_goals apply Tile.ext
  all_goals rintro ⟨i,j,⟨⟩⟩
  all_goals simp [Tile.cop, Tile.expandDim, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.ge, ComparableDType.le, RealMath.upperTriTile,
    Tile.mat, Tile.select]
  all_goals split_ifs <;> norm_num

theorem executes (p : Layout) (swapped : Bool) (u : BlockState) :
    stepStmts (initCode p swapped) u = some (initial p u) := by
  unfold initCode
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange p.BT) _ = some
    (⟨fun i : TileIndex [p.BT] => i.1.val⟩ : Tile .nat [p.BT]) from by rw [evalOp]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (triangular_eval p swapped _ (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BS] (.const 0)) _ =
    some (Tile.vec fun _ => some (0 : ℝ)) from by simp [evalOp_full, evalOp_const]; rfl))]
  rw [stepStmts.nil]
  rfl

theorem chunks_cover (p : Layout) (hBT : 0 < p.BT) : p.T ≤ numChunks p * p.BT := by
  unfold numChunks
  have hd := Nat.div_add_mod (p.T + p.BT - 1) p.BT
  have hm := Nat.mod_lt (p.T + p.BT - 1) hBT
  rw [Nat.mul_comm]
  omega

theorem invariant (p : Layout) (u : BlockState) (s o : RegionName) (a : Nat → ℝ)
    (hm : ∀ off, u.readMem s off = a off) (hBT : 0 < p.BT) :
    Holds u p (launch u) s o a (numChunks p) (initial p u) := by
  refine ⟨rfl, le_rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [initial, launch]
  · simp [initial, launch]
  · simp [initial]
  · simp [initial, RealLoop.carry, RealMath.carry, RealMath.suffixValue]
  · intro off; simpa [initial] using hm off
  · intro t j ha ht
    have hc := chunks_cover p hBT
    have := ha.1
    omega
  · intro r off hn; rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealInit
