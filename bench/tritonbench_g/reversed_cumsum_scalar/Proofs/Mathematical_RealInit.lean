import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Mathematical_RealLoop
import VeriTile.Triton.Semantics.ProgramRewrite

/-! Both sources execute the same initialization from arbitrary registers.
The initial zero carry is the empty padded suffix beyond the last chunk. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealInit
open VeriTile.Triton RealProgram RealLoop
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000

def initial (u : BlockState) : BlockState :=
  (u.setReg "i_bh" .nat [] (Tile.scalar (u.pids 0))).setReg "b_z" .real [] (Tile.scalar (some 0))

theorem executes (u : BlockState) : stepStmts initCode u = some (initial u) := by
  unfold initCode
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [] (.const 0)) _ =
    some (Tile.scalar (some (0 : ℝ))) from by simp [evalOp_full, evalOp_const]))]
  rw [stepStmts.nil]
  rfl

theorem chunks_cover (T BT : Nat) (hBT : 0 < BT) : T ≤ numChunks T BT * BT := by
  unfold numChunks
  have hd := Nat.div_add_mod (T + BT - 1) BT
  have hm := Nat.mod_lt (T + BT - 1) hBT
  rw [Nat.mul_comm]
  omega

theorem invariant (T BT : Nat) (u : BlockState) (s o : RegionName)
    (a : Nat → ℝ) (hm : ∀ off, u.readMem s off = a off) (hBT : 0 < BT) :
    Holds u T BT (u.pids 0) s o a (numChunks T BT) (initial u) := by
  refine ⟨rfl, le_rfl, ?_, ?_, ?_, ?_, ?_⟩
  · simp [initial]
  · simp [initial, accumulated, Formula.carry_end]
  · intro off
    simpa [initial] using hm off
  · intro t ht hl
    have hc := chunks_cover T BT hBT
    omega
  · intro r off hn
    rfl

#axiomsClean executes
#axiomsClean chunks_cover
#axiomsClean invariant
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealInit
