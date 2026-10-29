import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.SourceSafetyForward
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.SourceSafetyForward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem post_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (h : ForwardTile s α) (u : State α) (carry : u.regs .real [s.FBV,s.BK] "h"=some h)
    (static : KernelStatic.Forward s bh iv u)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    Safety.traceSafeList (engine M) bounds (first .forward s f scale).after u := by
  obtain ⟨head,key,value,mask⟩ := static
  cases hf : f.final <;> cases hb : f.vectorBeta
  all_goals
    simp only [first,source,Source.forward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,hf,hb,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=250000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,
      State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
      FP.Structural.store,integralFloorDiv,integralMod,head,key,value,mask,carry,
      KernelStatic.mask,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
  all_goals
    intro j i key value
    simpa [stateAddress,Nat.add_assoc] using access.terminal hf i j key value

theorem loop_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α)
    (inv : KernelForwardLoop.Holds M s f scale bh iv initial n u)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    Safety.rangeSafe (engine M) bounds "_i" n s.T 1 (first .forward s f scale).body u := by
  rw [Safety.rangeSafe_unfold,if_neg (by decide)]
  by_cases inside : n<s.T
  · rw [if_pos inside]
    have bound := KernelForwardLoop.bind_holds M s f scale bh iv initial n u inv
    refine ⟨body_safe M s f scale bh iv n inside _ _ bound.regs bounds access,?_⟩
    obtain ⟨v,ran,next⟩ := KernelForwardLoop.iteration M s f scale bh iv initial n _ inside bound
    rw [ran]
    exact loop_safe M s f scale bh iv initial (n+1) v next bounds access
  · rw [if_neg inside]
    trivial
termination_by s.T-n

theorem whole (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (u : State α) (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    TraceSafe (engine M) bounds (source .forward s f scale) u := by
  change Safety.traceSafeList (engine M) bounds (source .forward s f scale).surfaceBody u
  rw [source_body,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prefix_safe M s f scale bh iv u hv hk hh bounds access,?_⟩
  intro start ran
  have same : start=(KernelPrefix.forward M s f scale u).val := by
    simpa only [(KernelPrefix.forward M s f scale u).property,Option.some.injEq] using ran.symm
  subst start
  simp only [List.singleton_append,Safety.traceSafeList_cons]
  have inv := KernelForwardLoop.initial M s f scale bh iv u hv hk hh
  refine ⟨?_,?_⟩
  · simpa only [KernelProgram.loop,Safety.stepSafe_unfold] using
      loop_safe M s f scale bh iv u 0 _ inv bounds access
  · intro finish executed
    obtain ⟨middle,run,finalInv⟩ := KernelForwardLoop.executes M s f scale bh iv u 0 _ inv
    have stepEq : FP.Structural.step (engine M) (loop .forward s f scale)
        (KernelPrefix.forward M s f scale u).val=some middle := by
      simpa only [KernelProgram.loop,FP.Structural.step] using run
    have same : finish=middle := by simpa only [stepEq,Option.some.injEq] using executed.symm
    subst finish
    exact post_safe M s f scale bh iv _ middle finalInv.regs.carry finalInv.static bounds access

#axiomsClean post_safe
#axiomsClean loop_safe
#axiomsClean whole
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.SourceSafetyForward
