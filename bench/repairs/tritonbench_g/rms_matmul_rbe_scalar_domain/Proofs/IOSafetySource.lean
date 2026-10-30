import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOSafetyLoop
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOSafetyEpilogue

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem callee_partition (native : Bool) (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments) :
    calleeCode native plan fp8 outputFP8 p=prelude fp8 outputFP8 p ++
      [.forRangeDyn "step" (.constNat 0) (countExpr p) (.constNat 1) (IOBytes.code native fp8 outputFP8 p)] ++
      (if native then selectedEpilogue plan fp8 outputFP8 p else epilogue fp8 outputFP8 p) := by
  cases native
  · exact partition fp8 outputFP8 p
  · rfl

theorem callee_safe (a : Arithmetic α) (native : Bool) (plan : RotaryPlan)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (he : s.regs .real [] "EPS"=some (fun _=>epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _=>theta))
    (bytes : IOBytes.InputBytes fp8 p s) (capacity : RegionName → Nat) (bound : Bounds p s capacity) :
    traceSafeList a.model capacity (calleeCode native plan fp8 outputFP8 p) s := by
  let preState := prefixExecution a.model fp8 outputFP8 p s
  have pf : preState.val.mem=s.mem := prefix_frame a.model fp8 outputFP8 p s
  have pv := prefix_values a fp8 outputFP8 p s
  have pv0 : LoopRegisters (IOBytes.loop native a fp8 p s (initialValues a p s) 0) preState.val := by
    cases native <;> exact pv
  obtain ⟨last,hl,hv,hf,hh,hp⟩ := IOBytes.execution a native fp8 outputFP8 p s (initialValues a p s)
    bytes (IOBytes.initial_regions a p s) ((p.K+63)/64) 0 preState.val (by omega) pf pv0
  have ps := prefix_scalars a.model fp8 outputFP8 p s
  have pc := prefix_coordinates a.model fp8 outputFP8 p s
  have le : last.regs .real [] "EPS"=some (fun _ => epsilon) :=
    (hh _ _ _ (by simp [headerNames])).trans (ps.1.trans he)
  have lt : last.regs .real [] "THETA"=some (fun _ => theta) :=
    (hh _ _ _ (by simp [headerNames])).trans (ps.2.trans ht)
  have lb : last.regs .nat [] "pid_batch"=some (fun _ => s.pids 0) :=
    (hh _ _ _ (by simp [headerNames])).trans pc.1
  have lm : last.regs .nat [] "pid_m"=some (fun _ => programRow p s) :=
    (hh _ _ _ (by simp [headerNames])).trans pc.2.1
  have ln : last.regs .nat [] "pid_n"=some (fun _ => programCol p s) :=
    (hh _ _ _ (by simp [headerNames])).trans pc.2.2
  have safeLoop:=loop_safe a native fp8 outputFP8 p s bytes capacity bound
    ((p.K+63)/64) 0 preState.val (by omega) pf pv0
  have safeStep : stepSafe a.model capacity
      (.forRangeDyn "step" (.constNat 0) (countExpr p) (.constNat 1)
        (IOBytes.code native fp8 outputFP8 p)) preState.val := by
    simpa [stepSafe_unfold,safeOp_unfold,evalOp_unfold,countExpr,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex] using safeLoop
  have hstep : FP.Structural.step a.model
      (.forRangeDyn "step" (.constNat 0) (countExpr p) (.constNat 1)
        (IOBytes.code native fp8 outputFP8 p)) preState.val=some last := by
    simpa [FP.Structural.step,evalOp_unfold,countExpr,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex] using hl
  rw [callee_partition,List.append_assoc,traceSafeList_append]
  refine ⟨prefix_safe a.model fp8 outputFP8 p capacity s,?_⟩
  intro next ran
  have eq : next=preState.val := Option.some.inj (ran.symm.trans preState.property)
  subst next
  apply traceSafeList_cons_of_step safeStep hstep
  exact epilogue_safe a native plan fp8 outputFP8 p last epsilon theta
    (s.pids 0) (programRow p s) (programCol p s) _ _ le lt lb lm ln hv.moments hv.accumulator
    capacity bound.out

#axiomsClean callee_safe
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
