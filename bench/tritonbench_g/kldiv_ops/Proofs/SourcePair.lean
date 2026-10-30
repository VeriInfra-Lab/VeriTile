import bench.tritonbench_g.kldiv_ops.IO
import bench.tritonbench_g.kldiv_ops.Proofs.Mathematical_FP
import VeriTile.Triton.Float.StructuralSafety

/-! The existing signed output-row pointer candidate, transplanted to the current
runtime-epsilon forward and complete backward sources. No floating rewrite is used. -/
namespace VeriTile.Bench.Repairs.KldivOps.SourcePair
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety

noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward_decomposition (p : Parameters) (flag candidate : Bool) (red : Nat) :
    (if candidate then KernelDSL.forwardCandidate p flag red else KernelDSL.forward p flag red).surfaceBody =
      VeriTile.Bench.TritonBenchG.KldivOps.FPProof.forwardPrefix candidate p.Y p.T p.O p.cols p.cols (if red=0 then p.cols else 1) ++
        (KernelDSL.forward p flag red).surfaceBody.drop 4 := by
  cases flag <;> cases candidate <;> rfl

theorem backward_decomposition (p : Parameters) (flag candidate : Bool) :
    (if candidate then KernelDSL.backwardCandidate p flag else KernelDSL.backward p flag).surfaceBody =
      VeriTile.Bench.TritonBenchG.KldivOps.FPProof.backwardPrefix candidate p.T p.O p.cols p.cols ++
        (KernelDSL.backward p flag).surfaceBody.drop 3 := by
  cases flag <;> cases candidate <;> rfl

theorem forward_run (M : Algebra α) (p : Parameters) (flag : Bool) (red : Nat) (s : State α) :
    run (engine M) (KernelDSL.forwardCandidate p flag red).surfaceBody s =
      run (engine M) (KernelDSL.forward p flag red).surfaceBody s := by
  conv_lhs => rw [show (KernelDSL.forwardCandidate p flag red).surfaceBody = _ from
    forward_decomposition p flag Bool.true red]
  conv_rhs => rw [show (KernelDSL.forward p flag red).surfaceBody = _ from
    forward_decomposition p flag Bool.false red]
  exact (VeriTile.Bench.TritonBenchG.KldivOps.FPProof.forward_prefix_run M s _ _ _ _ _ _ _).symm

theorem backward_run (M : Algebra α) (p : Parameters) (flag : Bool) (s : State α) :
    run (engine M) (KernelDSL.backwardCandidate p flag).surfaceBody s =
      run (engine M) (KernelDSL.backward p flag).surfaceBody s := by
  conv_lhs => rw [show (KernelDSL.backwardCandidate p flag).surfaceBody = _ from
    backward_decomposition p flag Bool.true]
  conv_rhs => rw [show (KernelDSL.backward p flag).surfaceBody = _ from
    backward_decomposition p flag Bool.false]
  exact (VeriTile.Bench.TritonBenchG.KldivOps.FPProof.backward_prefix_run M s _ _ _ _ _).symm

private theorem pure_trace (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt) (safe : ∀ cmd ∈ code, ∀ s, stepSafe (engine M) bounds cmd s)
    (s : State α) : traceSafeList (engine M) bounds code s := by
  induction code generalizing s with
  | nil => exact traceSafeList_nil _ _ _
  | cons cmd rest ih =>
    apply traceSafeList_cons_intro (safe cmd (by simp) s)
    intro t _
    exact ih (fun next h => safe next (by simp [h])) t

private theorem prefix_safe (M : Algebra α) (bounds : RegionName → Nat)
    (Y T O : RegionName) (sy st so : Nat) (v : Bool) (s : State α) :
    traceSafeList (engine M) bounds (VeriTile.Bench.TritonBenchG.KldivOps.FPProof.forwardPrefix v Y T O sy st so) s ∧
      traceSafeList (engine M) bounds (VeriTile.Bench.TritonBenchG.KldivOps.FPProof.backwardPrefix v T O st so) s := by
  constructor <;> apply pure_trace <;> intro cmd member before
  all_goals simp only [VeriTile.Bench.TritonBenchG.KldivOps.FPProof.forwardPrefix,VeriTile.Bench.TritonBenchG.KldivOps.FPProof.backwardPrefix,List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals rcases member with rfl | rfl | rfl | rfl
  all_goals cases v <;> simp only [VeriTile.Bench.TritonBenchG.KldivOps.FPProof.rowPointer,Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    stepSafe_unfold,safeExpr_unfold,safeOp_unfold,true_and,and_true]

theorem forward_safe (M : Algebra α) (p : Parameters) (flag : Bool) (red : Nat)
    (bounds : RegionName → Nat) (s : State α) :
    TraceSafe (engine M) bounds (KernelDSL.forwardCandidate p flag red) s ↔
      TraceSafe (engine M) bounds (KernelDSL.forward p flag red) s := by
  unfold TraceSafe
  conv_lhs => rw [show (KernelDSL.forwardCandidate p flag red).surfaceBody = _ from
    forward_decomposition p flag Bool.true red]
  conv_rhs => rw [show (KernelDSL.forward p flag red).surfaceBody = _ from
    forward_decomposition p flag Bool.false red]
  rw [traceSafeList_append,traceSafeList_append]
  have runs := VeriTile.Bench.TritonBenchG.KldivOps.FPProof.forward_prefix_run M s p.Y p.T p.O p.cols p.cols (if red=0 then p.cols else 1) []
  simp only [List.append_nil] at runs
  simp only [(prefix_safe M bounds p.Y p.T p.O p.cols p.cols (if red=0 then p.cols else 1) Bool.true s).1,
    (prefix_safe M bounds p.Y p.T p.O p.cols p.cols (if red=0 then p.cols else 1) Bool.false s).1,
    true_and,runs]

theorem backward_safe (M : Algebra α) (p : Parameters) (flag : Bool)
    (bounds : RegionName → Nat) (s : State α) :
    TraceSafe (engine M) bounds (KernelDSL.backwardCandidate p flag) s ↔
      TraceSafe (engine M) bounds (KernelDSL.backward p flag) s := by
  unfold TraceSafe
  conv_lhs => rw [show (KernelDSL.backwardCandidate p flag).surfaceBody = _ from
    backward_decomposition p flag Bool.true]
  conv_rhs => rw [show (KernelDSL.backward p flag).surfaceBody = _ from
    backward_decomposition p flag Bool.false]
  rw [traceSafeList_append,traceSafeList_append]
  have runs := VeriTile.Bench.TritonBenchG.KldivOps.FPProof.backward_prefix_run M s p.T p.O p.cols p.cols []
  simp only [List.append_nil] at runs
  simp only [(prefix_safe M bounds p.Y p.T p.O p.cols p.cols p.cols Bool.true s).2,
    (prefix_safe M bounds p.Y p.T p.O p.cols p.cols p.cols Bool.false s).2,true_and,runs]

#axiomsClean forward_run
#axiomsClean backward_run
#axiomsClean forward_safe
#axiomsClean backward_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.SourcePair
