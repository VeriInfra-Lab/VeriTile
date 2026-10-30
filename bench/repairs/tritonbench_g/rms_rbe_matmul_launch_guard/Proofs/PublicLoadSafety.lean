import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicSourceSafety

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure LoadRegs (p : Parameters) (s : State α) (cur : Nat) (t : State α) : Prop where
  offk : t.regs .nat [64] "offs_k"=some (fun i => i.1.val)
  index : t.regs .nat [] "step"=some (fun _ => cur)
  xptr : t.regs .ptr [16,64] "x_ptrs"=some (xPointer p s cur)
  wptr : t.regs .ptr [64,64] "w_ptrs"=some (wPointer p s cur)
  rptr : t.regs .ptr [1,64] "rms_w_ptrs"=some (rPointer cur)

theorem loadRegs_ready (p : Parameters) (s t : State α) (cur : Nat) (acc : Accumulator α)
    (h : Ready p s t cur acc) : LoadRegs p s cur (t.setReg "step" .nat [] (fun _ => cur)) := by
  constructor <;> simp [State.setReg,h.offk,h.xptr,h.wptr,h.rptr]


theorem loadRegs_assign (a : Operations α) (p : Parameters) (s t u : State α) (cur : Nat)
    (h : LoadRegs p s cur t) (d sh name) (value : ComputeExpr d sh)
    (other : name∉["offs_k","step","x_ptrs","w_ptrs","rms_w_ptrs"])
    (ran : step (engine a) (.assign d sh name value) t=some u) : LoadRegs p s cur u := by
  simp only [step] at ran
  cases he : evalExpr (engine a) value t with
  | none => simp [he] at ran
  | some v =>
    simp only [he,bind,pure,Option.bind_some,Option.some.injEq] at ran
    subst u
    simp only [List.mem_cons,List.not_mem_nil,or_false,not_or] at other
    rcases other with ⟨h1,h2,h3,h4,h5⟩
    constructor <;> simp [State.setReg,Ne.symm h1,Ne.symm h2,Ne.symm h3,Ne.symm h4,Ne.symm h5,
      h.offk,h.index,h.xptr,h.wptr,h.rptr]


theorem trace_load_regs (a : Operations α) (bounds : RegionName → Nat)
    (p : Parameters) (s : State α) (cur : Nat) (code : List ComputeStmt)
    (safe : ∀ st ∈ code, ∀ t, LoadRegs p s cur t → Safety.stepSafe (engine a) bounds st t)
    (keep : ∀ st ∈ code, ∀ t u, LoadRegs p s cur t →
      step (engine a) st t=some u → LoadRegs p s cur u)
    (t : State α) (h : LoadRegs p s cur t) : Safety.traceSafeList (engine a) bounds code t := by
  induction code generalizing t with
  | nil => simp
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro
    · exact safe st (by simp) t h
    · intro u ran
      exact ih (fun st member => safe st (by simp [member]))
        (fun st member => keep st (by simp [member])) u (keep st (by simp) t u h ran)


def loadPrefix (fp8 : Bool) (p : Parameters) := (loop fp8 p).take (if fp8 then 12 else 7)
def pointerTail (fp8 : Bool) (p : Parameters) := (loop fp8 p).drop (if fp8 then 12 else 7)


theorem load_prefix_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (cur : Nat)
    (h : LoadRegs p s cur t) (backed : SourceBounds bounds p s) :
    Safety.traceSafeList (engine a) bounds (loadPrefix fp8 p) t := by
  apply trace_load_regs a bounds p s cur _ ?_ ?_ t h
  · intro st member u ready
    cases fp8 <;>
      simp only [loadPrefix,KernelSource.loop,kernel,TypedSource.kernel,RmsMatmulRbe.TypedSource.kernel,arguments,
        ComputeKernel.surfaceBody,Bool.false_eq_true,if_true,if_false,List.take,List.take_succ_cons,List.take_zero,List.drop,List.getElem?_cons_zero,List.getElem?_cons_succ,
        List.mem_cons,List.not_mem_nil,or_false] at member
    all_goals repeat' first | (rcases member with rfl | member) | (rcases member with rfl)
    all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      evalOp_unfold,Option.bind,numeric,bop,natLt,numericLt,Region.cast,
      ready.offk,ready.index,ready.xptr,ready.wptr,ready.rptr,xPointer,wPointer,rPointer,
      Broadcast.leftIndex,Broadcast.rightIndex,backed.x,backed.weight,backed.rms]
    all_goals first
      | exact fun i j b hi => backed.x cur (i,j,b) hi
      | exact fun i b hi => backed.rms cur (0,i,b) hi
      | exact fun i j b hi => backed.weight cur (i,j,b) hi
  · intro st member u v ready ran
    cases fp8 <;>
      simp only [loadPrefix,KernelSource.loop,kernel,TypedSource.kernel,RmsMatmulRbe.TypedSource.kernel,arguments,
        ComputeKernel.surfaceBody,Bool.false_eq_true,if_true,if_false,List.take,List.take_succ_cons,List.take_zero,List.drop,List.getElem?_cons_zero,List.getElem?_cons_succ,
        List.mem_cons,List.not_mem_nil,or_false] at member
    all_goals repeat' first | (rcases member with rfl | member) | (rcases member with rfl)
    all_goals exact loadRegs_assign a p s u v cur ready _ _ _ _ (by decide) ran

theorem pointer_tail_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (t : State α) :
    Safety.traceSafeList (engine a) bounds (pointerTail fp8 p) t := by
  cases fp8 <;>
    simp [pointerTail,KernelSource.loop,kernel,TypedSource.kernel,RmsMatmulRbe.TypedSource.kernel,
      arguments,ComputeKernel.surfaceBody,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem body_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (cur : Nat) (acc : Accumulator α)
    (h : Ready p s t cur acc) (backed : SourceBounds bounds p s) :
    Safety.traceSafeList (engine a) bounds (KernelSource.loop fp8 p)
      (t.setReg "step" .nat [] (fun _ => cur)) := by
  have split : KernelSource.loop fp8 p=loadPrefix fp8 p++pointerTail fp8 p :=
    (List.take_append_drop (if fp8 then 12 else 7) (KernelSource.loop fp8 p)).symm
  rw [split,Safety.traceSafeList_append]
  exact ⟨load_prefix_safe a fp8 p bounds s _ cur (loadRegs_ready p s t cur acc h) backed,
    fun u _ => pointer_tail_safe a fp8 p bounds u⟩

#axiomsClean load_prefix_safe
#axiomsClean body_safe
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
