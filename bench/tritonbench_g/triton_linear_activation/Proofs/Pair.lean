import bench.tritonbench_g.triton_linear_activation.Proofs.Support.PairIO
import bench.tritonbench_g.triton_linear_activation.Proofs.Support.CandidateKernels
import bench.tritonbench_g.triton_linear_activation.Proofs.IOBindings
import bench.tritonbench_g.triton_linear_activation.Proofs.KernelMemory
import bench.tritonbench_g.triton_linear_activation.Proofs.KernelContract
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Pair
open VeriTile Triton FP.Structural KernelSource KernelValues KernelContract
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 32000

/-- Inspect the common width prefix of the same copied candidate, with
unrelated constexpr flags fixed only for this proof projection. -/
def candidatePrefix (mode : Mode) (p : Parameters) : ComputeKernel :=
  CandidateSource.typed (dtype mode) p.output p.saved p.input p.weight p.bias
    p.rows p.columns p.inner p.outputRowStride p.outputColStride p.savedRowStride p.savedColStride
    p.inputRowStride p.inputInnerStride p.weightColStride p.weightInnerStride
    p.blockRows p.groupRows p.blockColumns p.blockInner Bool.false Bool.false Bool.false p.activation

def before (mode : Mode) (p : Parameters) := (flagged mode p Bool.false Bool.false Bool.false).surfaceBody.take 3
def after (mode : Mode) (p : Parameters) := (source mode p).surfaceBody.drop 4
def originalSite (mode : Mode) (p : Parameters) :=
  ((flagged mode p Bool.false Bool.false Bool.false).surfaceBody[3]? : Option ComputeStmt).getD (.effectMarker "unreachable")
def shiftedSite (mode : Mode) (p : Parameters) :=
  ((candidatePrefix mode p).surfaceBody[3]? : Option ComputeStmt).getD (.effectMarker "unreachable")

theorem original_code (mode : Mode) (p : Parameters) :
    (source mode p).surfaceBody = before mode p ++ originalSite mode p :: after mode p := by
  cases mode <;> cases hb : p.hasBias <;> cases hs : p.saveInputs <;>
    cases hl : (p.inner%p.blockInner==0) <;>
    simp only [before,after,originalSite,flagged,source,typed,originalKernel,dtype,hb,hs,hl,
      Bool.false_eq_true,if_false,if_true] <;> rfl

theorem shifted_code (mode : Mode) (p : Parameters) :
    (CandidateSource.source mode p).surfaceBody = before mode p ++ shiftedSite mode p :: after mode p := by
  cases mode <;> cases hb : p.hasBias <;> cases hs : p.saveInputs <;>
    cases hl : (p.inner%p.blockInner==0) <;>
    simp only [before,after,shiftedSite,candidatePrefix,flagged,source,typed,originalKernel,
      CandidateSource.source,CandidateSource.typed,candidateKernel,dtype,hb,hs,hl,
      Bool.false_eq_true,if_false,if_true] <;> rfl

theorem site_run {α : Type} [Inhabited α] (M : Algebra α) (mode : Mode) (p : Parameters) (s : State α) :
    step M (originalSite mode p) s = step M (shiftedSite mode p) s := by
  simp only [originalSite,shiftedSite,flagged,candidatePrefix,source,typed,originalKernel,CandidateSource.source,CandidateSource.typed,candidateKernel,
    Bool.false_eq_true,if_false,if_true,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  by_cases hg : p.groupRows=8
  all_goals simp only [step, FP.Structural.run, evalExpr, evalOp_unfold]
  all_goals cases hr : s.regs .nat [] "grid_n"
  all_goals simp [FP.Structural.run,step,evalExpr,evalOp_unfold,hr,hg,numeric,numericEq,FP.Structural.bop,
    Nat.shiftLeft_eq,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
  all_goals unfold FP.Structural.bop
  all_goals congr 1
  all_goals funext i
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]

theorem site_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (mode : Mode) (p : Parameters) (s : State α) :
    Safety.stepSafe M bounds (originalSite mode p) s ∧
      Safety.stepSafe M bounds (shiftedSite mode p) s := by
  simp only [originalSite,shiftedSite,flagged,candidatePrefix,source,typed,originalKernel,CandidateSource.source,CandidateSource.typed,candidateKernel,
    Bool.false_eq_true,if_false,if_true,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  by_cases hg : p.groupRows=8
  all_goals simp only [hg,decide_true,decide_false,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.traceSafeList_cons,Safety.traceSafeList_nil,evalOp_unfold,
    numericEq,Option.bind_some,bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    implies_true,and_true,true_and,Bool.false_eq_true,if_true,if_false]

theorem run_cons {α : Type} [Inhabited α] (M : Algebra α)
    (st : ComputeStmt) (rest : List ComputeStmt) (s : State α) :
    FP.Structural.run M (st::rest) s = (step M st s).bind (FP.Structural.run M rest) := by rw [FP.Structural.run]; rfl

theorem kernel_run {α : Type} [Inhabited α] (M : Algebra α) (mode : Mode) (p : Parameters) (s : State α) :
    exec M (source mode p) s = exec M (CandidateSource.source mode p) s := by
  change FP.Structural.run M (source mode p).surfaceBody s = FP.Structural.run M (CandidateSource.source mode p).surfaceBody s
  rw [original_code,shifted_code,run_append,run_append]
  apply congrArg (fun f => (FP.Structural.run M (before mode p) s).bind f)
  funext t
  rw [run_cons,run_cons,site_run]

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (mode : Mode) (p : Parameters) (s : State α) :
    TraceSafe M bounds (source mode p) s ↔ TraceSafe M bounds (CandidateSource.source mode p) s := by
  unfold TraceSafe
  rw [original_code,shifted_code,Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons]
  simp only [(site_safe M bounds mode p t).1,(site_safe M bounds mode p t).2,true_and,site_run]

theorem invocation_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (mode : Mode) (p : Parameters) (pid : Nat) (s : State α) :
    (LinearIO.invocation M D mode p pid).run s = (invocation M D mode p pid).run s := by
  change exec (engine M D) (source mode p) ((LinearIO.invocation M D mode p pid).initial s) =
    exec (engine M D) (CandidateSource.source mode p) ((LinearIO.invocation M D mode p pid).initial s)
  exact kernel_run _ mode p _

theorem invocation_safe {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (bounds : RegionName → Nat)
    (mode : Mode) (p : Parameters) (pid : Nat) (s : State α) :
    (LinearIO.invocation M D mode p pid).Safe bounds s ↔ (invocation M D mode p pid).Safe bounds s := by
  change TraceSafe (engine M D) bounds (source mode p) ((LinearIO.invocation M D mode p pid).initial s) ↔
    TraceSafe (engine M D) bounds (CandidateSource.source mode p) ((LinearIO.invocation M D mode p pid).initial s)
  exact kernel_safe _ bounds mode p _

theorem equivalent {α β : Type} [Inhabited β] (M : Algebra β) (D : Dot β)
    (encode : α → β) (observe : β → α) (mode : Mode) (p : Parameters) (pid : Nat)
    (positive : 0<p.blockInner) (strides : CanonicalStrides p) (separate : p.saved≠p.output) :
    (LinearIO.programIO M D encode observe mode p pid).Equivalent (variant M D encode observe mode p pid) := by
  intro vals _ bounds backed initial pinned
  have safe := LinearIO.schedule_safe M D encode observe mode p pid vals bounds backed initial positive
  let t := (LinearIO.invocation M D mode p pid).initial initial
  obtain ⟨final,ran,out,saved,exterior⟩ := KernelMemory.outputs M D mode p t positive separate
    (output_unique p t strides) (saved_unique p t strides)
  have frame := LinearIO.frame M D encode observe mode p pid vals initial final exterior
  refine ⟨safe,?_,final,final,?_,?_,?_,frame,frame⟩
  · change ScheduleSafe bounds [invocation M D mode p pid] initial
    rw [scheduleSafe_singleton,←invocation_safe]
    exact (scheduleSafe_singleton bounds _ initial).mp safe
  · change runSchedule [LinearIO.invocation M D mode p pid] initial=some final
    exact (runSchedule_singleton _ _).trans ran
  · change runSchedule [invocation M D mode p pid] initial=some final
    rw [runSchedule_singleton,←invocation_run]
    exact ran
  · intro o j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Pair
