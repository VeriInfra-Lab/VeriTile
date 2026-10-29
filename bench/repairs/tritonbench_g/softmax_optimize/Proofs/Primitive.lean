import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Source
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Recurrence
import bench.tritonbench_g.softmax_optimize.Proofs.SignedPrefix

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Primitive
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open SignedTransport
set_option maxHeartbeats 3000000
noncomputable section
local instance : DecidableEq (WithBot ℝ) := Classical.decEq _

/-- Exact execution of the widened signed address statements, from the actual
source's two scalar integer registers. No successful-execution premise. -/
theorem prefix_run (X : RegionName) (N B start row : Nat) (s : BlockState)
    (hpid : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hcols : s.regs .int [] "N" = some (Tile.scalar (N : Int))) :
    stepStmts (Source.indexPrefix X B) (s.setReg "start_n" .int [] (Tile.scalar (start : Int))) =
      some (prefixIntState X N B start row s) := by
  simp only [Source.indexPrefix]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := castIndex (columns B start)) (by
    simp only [evalOp.eq_def, BlockState.setReg_same, Option.bind_some, Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro lane
    change (start : Int) + (lane.1.val : Int) = ((start + lane.1.val : Nat) : Int)
    exact (Int.natCast_add _ _).symm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := castIndex (offsets N B start row)) (by
    simp only [evalOp.eq_def, BlockState.setReg_ne_name, BlockState.setReg_same, hpid, hcols,
      ne_eq, String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro lane
    change (row : Int) * (N : Int) + ((start + lane.1.val : Nat) : Int) =
      ((row * N + (start + lane.1.val) : Nat) : Int)
    simp only [Int.natCast_add, Int.natCast_mul]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := pointers X N B start row) (by
    simp only [evalOp.eq_def, BlockState.setReg_same, Option.bind_some, Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro lane
    change (X, 0 + Int.toNat ((row * N + (start + lane.1.val) : Nat) : Int)) =
      (X, row * N + (start + lane.1.val))
    rw [Int.toNat_natCast, Nat.zero_add]))]
  rw [stepStmts.nil]
  rfl

def newMaximum {B : Nat} (m x : Tile .real [B]) : Tile .real [B] :=
  ⟨fun i => m.data i ⊔ x.data i⟩

def oldMass {B : Nat} (m z x : Tile .real [B]) : Tile .real [B] :=
  ⟨fun i => if (m.data i ⊔ x.data i) = ⊥ then z.data i
    else WithBot.realMul (WithBot.realExp (WithBot.realSub (m.data i) (m.data i ⊔ x.data i))) (z.data i)⟩

def newMass {B : Nat} (m x : Tile .real [B]) : Tile .real [B] :=
  ⟨fun i => if x.data i = ⊥ then (0 : ℝ)
    else WithBot.realExp (WithBot.realSub (x.data i) (m.data i ⊔ x.data i))⟩

def newDenominator {B : Nat} (m z x : Tile .real [B]) : Tile .real [B] :=
  ⟨fun i => WithBot.realAdd ((oldMass m z x).data i) ((newMass m x).data i)⟩

def updateState {B : Nat} (s : BlockState) (m z x : Tile .real [B]) : BlockState :=
  (((((s.setReg "new_m" .real [B] (newMaximum m x)).setReg
    "old_mass" .real [B] (oldMass m z x)).setReg
    "new_mass" .real [B] (newMass m x)).setReg
    "new_z" .real [B] (newDenominator m z x)).setReg
    "m" .real [B] (newMaximum m x)).setReg
    "z" .real [B] (newDenominator m z x)

theorem update_run {B : Nat} (s : BlockState) (m z x : Tile .real [B])
    (hm : s.regs .real [B] "m" = some m)
    (hz : s.regs .real [B] "z" = some z)
    (hx : s.regs .real [B] "inp" = some x) :
    stepStmts (Source.update B) s = some (updateState s m z x) := by
  unfold Source.update SignedTransport.sourceMaximum
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := newMaximum m x) (by
    simp only [evalOp.eq_def, hm, hx, Option.bind_some, Option.bind_eq_bind]
    rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := oldMass m z x) (by
    simp only [evalOp.eq_def, BlockState.setReg_ne_name, BlockState.setReg_same, hm, hz, hx,
      ne_eq, String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro i
    simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.uop_data,
      Tile.scalar_data, ComparableDType.eq, decide_eq_true_eq, oldMass, newMaximum]
    rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := newMass m x) (by
    simp only [evalOp.eq_def, BlockState.setReg_ne_name, BlockState.setReg_same, hm, hz, hx,
      ne_eq, String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro i
    simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.uop_data,
      Tile.scalar_data, ComparableDType.eq, decide_eq_true_eq, newMass, newMaximum]
    norm_num <;> rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := newDenominator m z x) (by
    simp only [evalOp.eq_def, BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
    rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := newMaximum m x) (by
    simp only [evalOp_ref, BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := newDenominator m z x) (by
    simp only [evalOp_ref, BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]))]
  rw [stepStmts.nil]
  rfl

theorem update_values {B : Nat} (m z x : Tile .real [B]) (i : TileIndex [B]) :
    ((newMaximum m x).data i, (newDenominator m z x).data i) =
      Recurrence.step Recurrence.realOperations (m.data i, z.data i) (x.data i) := by
  simp only [newMaximum, newDenominator, oldMass, newMass,
    Recurrence.step, Recurrence.realOperations, decide_eq_true_eq]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.Primitive
