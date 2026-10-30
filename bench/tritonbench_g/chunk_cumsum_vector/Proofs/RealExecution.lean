import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealLoop
import bench.tritonbench_g.chunk_cumsum_vector.Sources.Mathematical_RealInit
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealExecution
open VeriTile.Triton IndexProgram RealProgram RealValues RealData RealLoop
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def initial (h : Call) (c : Config) (p : Geometry.Program h) (u : BlockState) : BlockState :=
  let u1 := u.setReg "i_s" .nat [] (Tile.scalar p.feature)
  let u2 := u1.setReg "i_bh" .nat [] (Tile.scalar p.row)
  let u3 := u2.setReg "i_b" .nat [] (Tile.scalar (p.row/h.H))
  let u4 := u3.setReg "i_h" .nat [] (Tile.scalar (p.row%h.H))
  let u5 := u4.setReg "o_i" .nat [c.BT] (lanes c.BT)
  let u6 := u5.setReg "m_s" .real [c.BT,c.BT] (lowerTriTile c.BT)
  let u7 := u6.setReg "b_z" .real [BS] (Tile.vec fun _ => some 0)
  let u8 := u7.setReg "start_s" .nat [] (Tile.scalar (p.feature*BS))
  let u9 := u8.setReg "lanes_s" .nat [BS] (lanes BS)
  let u10 := u9.setReg "active_s" .bool [BS] (axisMask h.S BS (p.feature*BS))
  u10.setReg "columns" .nat [BS] (coordinate h.S BS (p.feature*BS))

theorem init_executes (h : Call) (c : Config) (p : Geometry.Program h) (u : BlockState)
    (hf : u.pids 0=p.feature) (hr : u.pids 1=p.row) :
    stepStmts (initCode h c) u=some (initial h c p u) := by
  unfold initCode
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.programId 0) u=some (Tile.scalar p.feature) from by simp [evalOp,hf]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.programId 1) _=some (Tile.scalar p.row) from by simp [evalOp,hr]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.floorDiv .nat .nil (.ref .nat [] "i_bh") (.constNat h.H)) _=some (Tile.scalar (p.row/h.H)) from by simp [evalOp]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.mod .nat .nil (.ref .nat [] "i_bh") (.constNat h.H)) _=some (Tile.scalar (p.row%h.H)) from by simp [evalOp]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange c.BT) _=some (lanes c.BT) from by simpa [lanes,Tile.vec] using evalOp_arange c.BT _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (triangularOp c) _=some (lowerTriTile c.BT) from by
      exact VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealInit.triangular_eval
        ⟨0,0,0,h.T,h.S,c.BT,BS⟩ _ (by simp [lanes,Tile.vec])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [BS] (.const 0)) _=some (Tile.vec fun _ => some (0:ℝ)) from by simp [evalOp_full,evalOp_const]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (start_eval "i_s" BS p.feature _ (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange BS) _=some (lanes BS) from by simpa [lanes,Tile.vec] using evalOp_arange BS _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (active_eval h.S BS (p.feature*BS) "lanes_s" "start_s" _ (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (coordinate_eval h.S BS (p.feature*BS) "start_s" "active_s" "lanes_s" _ (by simp) (by simp) (by simp)))]
  rw [stepStmts.nil]
  rfl

theorem invariant (h : Call) (c : Config) (p : Geometry.Program h) (u : BlockState) :
    Holds u h c p (u.readMem h.input) 0 (initial h c p u) := by
  refine ⟨rfl,Nat.zero_le _,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [initial]
  · simp [initial]
  · simp [initial]
  · simp [initial]
  · simp [initial]
  · simp [initial,carry,VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealMath.carry]
  · intro off; simp [initial]
  · intro t j ha ht; simp at ht
  · intro r off hn; rfl

theorem layout_unique (h : Call) (p : Geometry.Program h) (layout : Geometry.OutputLayout h) :
    NoCollision h p := by
  intro t u i j ht hu heq
  obtain ⟨_,he,ho⟩ := layout.unique p.row p.row t u (p.feature*BS+i.val) (p.feature*BS+j.val)
    p.rowInside p.rowInside ht.1 hu.1 ht.2 hu.2 heq
  exact ⟨he,Fin.ext (by omega)⟩

/-- Complete production real projection: independent logical prefix values,
arbitrary padding, successful execution and a full memory-cell frame. -/
theorem source_executes (h : Call) (c : Config) (hc : c∈configurations)
    (p : Geometry.Program h) (layout : Geometry.OutputLayout h) (u : BlockState)
    (hf : u.pids 0=p.feature) (hr : u.pids 1=p.row) :
    ∃ v, exec (Kernels.source h c).eraseDType.toAlgKernel u=some v ∧
      (∀ (t : Nat) (j : Fin BS), active h p t j →
        v.readMem h.output (address h p t j)=expected h p.feature p.row (u.readMem h.input) t j) ∧
      BlockState.WriteWithin (footprint h p) u v := by
  have positive := (Kernels.configurations_positive c hc).1
  obtain ⟨v,he,inv⟩ := RealLoop.executes u h c positive p (u.readMem h.input) layout.distinct
    (layout_unique h p layout) (initial h c p u) (invariant h c p u)
  refine ⟨v,?_,?_,inv.frame⟩
  · rw [exec,source_body,body,ProgramRewrite.append,init_executes h c p u hf hr]
    change stepStmts [statement h c] (initial h c p u)=some v
    rw [stepStmts.cons_some he,stepStmts.nil]
  · intro t j ha
    exact inv.output t j ha (ha.1.trans_le (Geometry.ceiling_covers h.T c.BT positive))

#axiomsClean source_executes
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealExecution
