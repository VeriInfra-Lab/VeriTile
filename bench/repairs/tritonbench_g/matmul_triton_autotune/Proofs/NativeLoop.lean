import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.NativeControl

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
open VeriTile.Triton FP.Structural Whole
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 2000000

/-- An explicit finite-word counted loop. The guard and backedge use signed
i64 comparison and modular addition. Fuel merely makes this partial evaluator
total; exhaustion while the source guard is true is failure, never success. -/
def loop {β : Type} (step : BitVec 64 → β → β) :
    Nat → BitVec 64 → BitVec 64 → β → Option (BitVec 64 × β)
  | 0,cur,stop,acc => if cur.slt stop then none else some (cur,acc)
  | fuel+1,cur,stop,acc => if cur.slt stop then
      loop step fuel (cur+word 1) stop (step cur acc) else some (cur,acc)

theorem loop_correspondence {β : Type} (step : BitVec 64 → β → β) (update : Nat → β → β)
    (stop : Nat) (hstop : stop<2^63)
    (body : ∀ k, k<stop → step (word k)=update k)
    (fuel cur : Nat) (hcur : cur≤stop) (enough : stop-cur≤fuel) (acc : β) :
    loop step fuel (word cur) (word stop) acc=
      some (word stop,foldRange update cur stop 1 acc) := by
  induction fuel generalizing cur acc with
  | zero =>
    have eq : cur=stop := by omega
    subst cur
    simp [loop,slt_eq stop stop hstop hstop,foldRange]
  | succ fuel ih =>
    by_cases h : cur<stop
    · rw [loop,slt_eq cur stop (by omega) hstop]
      simp only [h,decide_true,if_true]
      rw [← BitVec.ofNat_add,body cur h,ih (cur+1) (by omega) (by omega)]
      conv_rhs => rw [foldRange]
      simp [h]
    · have eq : cur=stop := by omega
      subst cur
      simp [loop,slt_eq stop stop hstop hstop,foldRange]

theorem stop_bound {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) : r.parameters.numKBlocks<2^63 := by
  have h := (control_machine r s t pid hp).2.2.1
  exact lt_of_le_of_lt (Nat.div_le_self _ _) (by omega)

/-- Every induction value and the increment after the final body are exact.
The loop result above additionally proves the native guard stops at that value. -/
theorem backedge_exact {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid k : Nat) (hp : pid<programs r) (hk : k<r.parameters.numKBlocks) :
    (word k).toInt=(k : Int) ∧ (word k+word 1).toInt=((k+1 : Nat) : Int) := by
  have h := stop_bound r s t pid hp
  refine ⟨signed k (by omega),?_⟩
  rw [← BitVec.ofNat_add,signed (k+1) (by omega)]

#axiomsClean loop_correspondence
#axiomsClean backedge_exact
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
