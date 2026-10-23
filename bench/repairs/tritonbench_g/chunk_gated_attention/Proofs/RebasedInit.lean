import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RebasedLoop
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.Grid
import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHInit

/-! Flattened launch decoder and the actual zero-plus-seed initialization.
The old mathematical starting-state and invariant definitions are reused. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedInit
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles RealHLoads
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def decoded (p : Layout) (base : Nat) (u : BlockState) : Launch :=
  let linear := u.pids 0+base
  ⟨linear % (p.V/p.BV), (linear / (p.V/p.BV)) % (p.K/p.BK),
    linear / ((p.V/p.BV)*(p.K/p.BK))⟩

def registers (p : Layout) (base : Nat) (u : BlockState) : BlockState :=
  let l := decoded p base u
  ((((u.setReg "linear" .nat [] (Tile.scalar (u.pids 0+base))).setReg
    "i_v" .nat [] (Tile.scalar l.value)).setReg
    "i_k" .nat [] (Tile.scalar l.key)).setReg
    "i_bh" .nat [] (Tile.scalar l.head)).setReg
    "b_h" .real [p.BK,p.BV] (stateTile p (fun _ _ => 0))

def initial (p : Layout) (r : Regions) (a : Nat → ℝ) (base : Nat)
    (hasSeed : Bool) (u : BlockState) : BlockState :=
  if hasSeed then
    ((registers p base u).setReg "p_h" .blockPtr [p.BK,p.BV]
      (RebasedAccess.statePointer p (decoded p base u) r.seed)).setReg
      "b_h" .real [p.BK,p.BV] (stateTile p (seedData p (decoded p base u) a (fun _ => 0)))
  else registers p base u

theorem seed_load (p : Layout) (l : Launch) (r : RegionName) (a : Nat → ℝ) (u : BlockState)
    (hk : l.key*p.BK+p.BK ≤ p.K) (hv : l.value*p.BV+p.BV ≤ p.V)
    (hm : ∀ off, u.readMem r off = a off)
    (hp : u.regs .blockPtr [p.BK,p.BV] "p_h" = some (RebasedAccess.statePointer p l r)) :
    evalOp (load2 p.BK p.BV "p_h") u = some (stateTile p (seedData p l a (fun _ => 0))) := by
  rw [RebasedAccess.matrix_load r u "p_h" _ _ _ _ _ hp]
  congr 1
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  have he : l.key*p.BK+e.val < p.K := by omega
  have hj : l.value*p.BV+j.val < p.V := by omega
  simp only [stateTile,Tile.mat,RealHLoads.seedData,keyIndex,valueIndex,he,hj,and_self,if_true,hm,
    stateAddress,Nat.mul_one]
  congr 2 <;> ring

theorem add_seed (p : Layout) (l : Launch) (r : RegionName) (a : Nat → ℝ) (u : BlockState)
    (hk : l.key*p.BK+p.BK ≤ p.K) (hv : l.value*p.BV+p.BV ≤ p.V)
    (hm : ∀ off, u.readMem r off = a off)
    (hp : u.regs .blockPtr [p.BK,p.BV] "p_h" = some (RebasedAccess.statePointer p l r))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (fun _ _ => 0))) :
    evalOp (.add .real (.consSame (.consSame .nil)) (carry p) (load2 p.BK p.BV "p_h")) u =
      some (stateTile p (seedData p l a (fun _ => 0))) := by
  rw [evalOp_add,carry,evalOp_ref,hh,seed_load p l r a u hk hv hm hp]
  simp only [Option.bind_eq_bind,Option.bind_some]
  congr 1
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp [stateTile,Tile.mat,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add]

theorem executes (p : Layout) (r : Regions) (a : Nat → ℝ) (base : Nat)
    (hasSeed : Bool) (u : BlockState)
    (hk : (decoded p base u).key*p.BK+p.BK ≤ p.K)
    (hv : (decoded p base u).value*p.BV+p.BV ≤ p.V)
    (hm : ∀ off, u.readMem r.seed off = a off) :
    stepStmts (RebasedProgram.initialization p r base hasSeed) u =
      some (initial p r a base hasSeed u) := by
  unfold RebasedProgram.initialization
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show
    evalOp (RebasedProgram.plus (.programId 0) (.constNat base)) u =
      some (Tile.scalar (u.pids 0+base)) from by
        simp only [RebasedProgram.plus,evalOp_add,evalOp_programId,evalOp_constNat]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show
    evalOp (.mod .nat .nil (.ref .nat [] "linear") (.constNat (p.V/p.BV))) _ =
      some (Tile.scalar (decoded p base u).value) from by
        simp only [evalOp,BlockState.setReg_same,Option.bind_eq_bind,Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show
    evalOp (.mod .nat .nil (.div .nat .nil (.ref .nat [] "linear") (.constNat (p.V/p.BV)))
      (.constNat (p.K/p.BK))) _ = some (Tile.scalar (decoded p base u).key) from by
        simp only [evalOp,BlockState.setReg_same,BlockState.setReg_ne_name,Option.bind_eq_bind,Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show
    evalOp (.div .nat .nil (.ref .nat [] "linear") (.constNat ((p.V/p.BV)*(p.K/p.BK)))) _ =
      some (Tile.scalar (decoded p base u).head) from by
        simp only [evalOp,BlockState.setReg_same,BlockState.setReg_ne_name,Option.bind_eq_bind,Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BK,p.BV] (.const 0)) _ =
    some (stateTile p (fun _ _ => 0)) from by simp [evalOp_full,evalOp_const]; rfl))]
  cases hasSeed
  · rw [stepStmts.cons_some (RealHInit.if_false _ _),stepStmts.nil]
    rfl
  · rw [stepStmts.cons_some (st := .ifThen (.constBool Bool.true) _) (by
      rw [RealHInit.if_true]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (RebasedAccess.state_eval p (decoded p base u) r.seed _
        (by simp) (by simp) (by simp)))]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (add_seed p (decoded p base u) r.seed a _ hk hv
        (by simpa using hm) (by simp) (by simp))),stepStmts.nil])]
    rw [stepStmts.nil]
    rfl

theorem invariant (p : Layout) (r : Regions) (a : Inputs) (base : Nat)
    (gateK hasSeed : Bool) (u : BlockState)
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hg : ∀ off, u.readMem r.gate off = a.gate off) :
    RealHLoop.Holds u p (decoded p base u) r a gateK hasSeed 0 (initial p r a.seed base hasSeed u) := by
  refine ⟨Nat.zero_le _,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · cases hasSeed <;> simp [initial,registers]
  · cases hasSeed <;> simp [initial,registers]
  · cases hasSeed <;> simp [initial,registers]
  · rw [← RealHInit.starting_expected]
    cases hasSeed <;> simp [initial,registers,RealHInit.starting]
  · intro off; cases hasSeed <;> simpa [initial,registers] using hk off
  · intro off; cases hasSeed <;> simpa [initial,registers] using hv off
  · intro off; cases hasSeed <;> simpa [initial,registers] using hg off
  · intro q _ _ hq; omega
  · intro _ _ _; cases hasSeed <;> rfl
  · cases hasSeed <;> rfl
  · cases hasSeed <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedInit
