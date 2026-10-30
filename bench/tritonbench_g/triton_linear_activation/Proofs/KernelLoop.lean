import bench.tritonbench_g.triton_linear_activation.Proofs.KernelPrefix

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelLoop
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelPrefix
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def mirror (p : Parameters) (cur : Nat) : Int := (cur*p.blockInner:Int)-(p.inner:Int)

theorem body_run (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (s t : State α) (cur : Nat) (a : Accumulator α p) (h : Ready p s t cur a) :
    ∃ u, run (engine M D) (body mode p)
      (t.setReg "__rev_k" .int [] (fun _ => mirror p cur))=some u ∧
      Ready p s u (cur+1) (update M D mode p s cur a) := by
  cases mode <;> by_cases mask : p.inner%p.blockInner=0
  all_goals
    simp [body,statement,source,typed,dtype,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,integralFloorDiv,integralMod,
      Region.cast,ofFloat,toFloat,State.setReg,h.mem,h.pids,h.undef,
      h.rawm,h.rawn,h.offm,h.offn,h.offk,h.aptr,h.bptr,h.acc,
      engine,Algebra.withDefaultPrecision,resolvePrecision,mask,mirror]
  all_goals constructor
  all_goals first
    | (constructor <;> simp [State.setReg,h.mem,h.pids,h.undef,h.rawm,h.rawn,h.offm,h.offn,h.offk])
    | (simp [State.setReg,h.mem,h.pids,h.undef,h.rawm,h.rawn,h.offm,h.offn,h.offk,h.aptr,h.bptr,h.acc,
        update,aValues,bValues,KernelValues.read,dtype,mask,mirror,paddedZero,
        aPointer,bPointer,aOffset,bOffset,bop,ofFloat,toFloat,Region.cast])
  all_goals first | rfl | (funext i; simp [aPointer,bPointer,aOffset,bOffset,bop,Nat.add_mul,Nat.add_assoc])
  all_goals
    rcases i with ⟨i,j,⟨⟩⟩
    unfold KernelValues.update KernelValues.aValues KernelValues.bValues
    simp [KernelValues.read,paddedZero,dtype,aOffset,bOffset,mask,ofFloat,toFloat,
      Nat.add_assoc,Nat.cast_mul]

theorem statement_eval (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters) (s : State α) :
    step (engine M D) (statement mode p) s=
      rangeInt (engine M D) "__rev_k" (mirror p 0) 0 p.blockInner (body mode p) s := by
  simp [statement,body,source,typed,ComputeKernel.surfaceBody,step,evalOp_unfold,mirror,
    numeric,bop,integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex]

theorem loop_run (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (s t : State α) (positive : 0<p.blockInner) (cur : Nat)
    (bound : cur≤ceilDiv p.inner p.blockInner)
    (h : Ready p s t cur (foldN (update M D mode p s) cur (initial M mode p s))) :
    ∃ u, rangeInt (engine M D) "__rev_k" (mirror p cur) 0 p.blockInner (body mode p) t=some u ∧
      Ready p s u (ceilDiv p.inner p.blockInner) (accumulated M D mode p s) := by
  by_cases inside : cur<ceilDiv p.inner p.blockInner
  · have upper := ceil_upper p.inner p.blockInner positive
    have product := Nat.mul_le_mul_right p.blockInner (Nat.succ_le_of_lt inside)
    have active : mirror p cur<0 := by simp only [mirror]; push_cast; nlinarith
    obtain ⟨v,he,hv⟩ := body_run M D mode p s t cur _ h
    have next : mirror p cur+(p.blockInner:Int)=mirror p (cur+1) := by simp [mirror,Nat.add_mul]; ring
    have nextReady : Ready p s v (cur+1) (foldN (update M D mode p s) (cur+1) (initial M mode p s)) := by
      simpa only [foldN] using hv
    obtain ⟨u,hu,final⟩ := loop_run M D mode p s v positive (cur+1) (by omega) nextReady
    refine ⟨u,?_,final⟩
    rw [rangeInt]
    simp only [Nat.ne_of_gt positive,if_false,active,if_true,he,bind,Option.bind_some,next]
    exact hu
  · have last : cur=ceilDiv p.inner p.blockInner := by omega
    have cover := ceil_cover p.inner p.blockInner positive
    have inactive : ¬mirror p cur<0 := by simp only [mirror,last]; push_cast; omega
    refine ⟨t,?_,?_⟩
    · rw [rangeInt]
      simp [Nat.ne_of_gt positive,inactive]
    · simpa only [last,accumulated] using h
termination_by ceilDiv p.inner p.blockInner-cur

#axiomsClean body_run
#axiomsClean statement_eval
#axiomsClean loop_run
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelLoop
