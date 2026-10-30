import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.CandidateKernels
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.SourcePair
open VeriTile Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def before (p : Parameters) (st : Storage) := (Source.typed p st).surfaceBody.take 7
def after (p : Parameters) (st : Storage) := (Source.typed p st).surfaceBody.drop 8
def originalSite (p : Parameters) (st : Storage) :=
  ((Source.typed p st).surfaceBody[7]? : Option ComputeStmt).getD (.effectMarker "unreachable")
def shiftedSite (p : Parameters) (st : Storage) :=
  ((CandidateSource.typed p st).surfaceBody[7]? : Option ComputeStmt).getD (.effectMarker "unreachable")

theorem original_code (p : Parameters) (st : Storage) :
    (Source.typed p st).surfaceBody = before p st ++ originalSite p st :: after p st := by
  cases st <;> rfl

theorem shifted_code (p : Parameters) (st : Storage) :
    (CandidateSource.typed p st).surfaceBody = before p st ++ shiftedSite p st :: after p st := by
  cases st <;> rfl

theorem site_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (st : Storage) (s : State α) :
    step M (originalSite p st) s = step M (shiftedSite p st) s := by
  cases st <;>
    simp only [originalSite,shiftedSite,Source.typed,Source.halfSource,Source.byteSource,CandidateSource.typed,
    CandidateSource.halfSource,CandidateSource.byteSource,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  all_goals by_cases hg : p.GROUP_SIZE_M=8
  all_goals try simp only [hg,decide_true,decide_false]
  all_goals simp only [step, FP.Structural.run, evalExpr, evalOp_unfold]
  all_goals cases hr : s.regs .int [] "num_pid_n"
  all_goals simp [FP.Structural.run,step,evalExpr,evalOp_unfold,hr,hg,numeric,numericEq,FP.Structural.bop,
    Op.intShiftLeft,IntegerExpr.sameShape,Nat.shiftLeft_eq,Broadcast.leftIndex,Broadcast.rightIndex,Int.mul_comm]
  all_goals unfold FP.Structural.bop
  all_goals congr 1
  all_goals funext i
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,Int.mul_comm]

theorem site_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (p : Parameters) (st : Storage) (s : State α) :
    Safety.stepSafe M bounds (originalSite p st) s ∧
      Safety.stepSafe M bounds (shiftedSite p st) s := by
  cases st <;>
    simp only [originalSite,shiftedSite,Source.typed,Source.halfSource,Source.byteSource,CandidateSource.typed,
    CandidateSource.halfSource,CandidateSource.byteSource,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  all_goals by_cases hg : p.GROUP_SIZE_M=8
  all_goals simp only [hg,decide_true,decide_false,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.traceSafeList_cons,Safety.traceSafeList_nil,evalOp_unfold,Op.intShiftLeft,
    implies_true,and_true,true_and,Bool.false_eq_true,if_true,if_false,
    numericEq,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

  all_goals simp [Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,hg]

theorem run_cons {α : Type} [Inhabited α] (M : Algebra α)
    (st : ComputeStmt) (rest : List ComputeStmt) (s : State α) :
    FP.Structural.run M (st::rest) s = (step M st s).bind (FP.Structural.run M rest) := by rw [FP.Structural.run]; rfl

theorem kernel_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (st : Storage) (s : State α) :
    exec M (Source.typed p st) s = exec M (CandidateSource.typed p st) s := by
  change FP.Structural.run M (Source.typed p st).surfaceBody s = FP.Structural.run M (CandidateSource.typed p st).surfaceBody s
  rw [original_code,shifted_code,run_append,run_append]
  apply congrArg (fun f => (FP.Structural.run M (before p st) s).bind f)
  funext t
  rw [run_cons,run_cons,site_run]

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (p : Parameters) (st : Storage) (s : State α) :
    TraceSafe M bounds (Source.typed p st) s ↔ TraceSafe M bounds (CandidateSource.typed p st) s := by
  unfold TraceSafe
  rw [original_code,shifted_code,Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons]
  simp only [(site_safe M bounds p st t).1,(site_safe M bounds p st t).2,true_and,site_run]


#axiomsClean kernel_run
#axiomsClean kernel_safe
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.SourcePair
