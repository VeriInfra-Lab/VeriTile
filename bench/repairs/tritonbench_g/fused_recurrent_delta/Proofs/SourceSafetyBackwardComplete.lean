import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.SourceSafetyBackward
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.SourceSafetyBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
variable {α : Type} [Inhabited α]

theorem reverse_loop_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α)
    (inv : KernelReverseLoop.Holds M s f scale bh iv initial n u)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    Safety.rangeSafe (engine M) bounds "_i" n s.T 1 (first .backward s f scale).body u := by
  rw [Safety.rangeSafe_unfold,if_neg (by decide)]
  by_cases inside : n<s.T
  · rw [if_pos inside]
    have bound := KernelReverseLoop.bind_holds M s f scale bh iv initial n u inv
    refine ⟨reverse_safe M s f scale bh iv (s.T-1-n) (by omega) _ _ (bound.regs inside) bounds access,?_⟩
    obtain ⟨v,ran,next⟩ := KernelReverseLoop.iteration M s f scale bh iv initial n _ inside bound
    rw [ran]
    exact reverse_loop_safe M s f scale bh iv initial (n+1) v next bounds access
  · rw [if_neg inside]
    trivial
termination_by s.T-n

theorem recompute_loop_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α)
    (inv : KernelRecomputeLoop.Holds M s f scale bh iv initial n u)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    Safety.rangeSafe (engine M) bounds "i" n s.T 1 (second s f scale).body u := by
  rw [Safety.rangeSafe_unfold,if_neg (by decide)]
  by_cases inside : n<s.T
  · rw [if_pos inside]
    have bound := KernelRecomputeLoop.bind_holds M s f scale bh iv initial n u inv
    refine ⟨recompute_safe M s f scale bh iv n inside _ _ bound.regs bounds access,?_⟩
    obtain ⟨v,ran,next⟩ := KernelRecomputeLoop.iteration M s f scale bh iv initial n _ inside bound
    rw [ran]
    exact recompute_loop_safe M s f scale bh iv initial (n+1) v next bounds access
  · rw [if_neg inside]
    trivial
termination_by s.T-n

theorem whole (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (u : State α) (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    TraceSafe (engine M) bounds (source .backward s f scale) u := by
  change Safety.traceSafeList (engine M) bounds (source .backward s f scale).surfaceBody u
  rw [source_body,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prefix_safe M s f scale bh iv u hv hk hh bounds access,?_⟩
  intro pre ran
  have same : pre=(KernelPrefix.backward M s f scale u).val := by
    simpa only [(KernelPrefix.backward M s f scale u).property,Option.some.injEq] using ran.symm
  subst pre
  simp only [List.singleton_append,Safety.traceSafeList_cons]
  have inv := KernelReverseLoop.initial M s f scale bh iv u hv hk hh
  refine ⟨?_,?_⟩
  · simpa [KernelProgram.loop,Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold] using
      reverse_loop_safe M s f scale bh iv u 0 _ inv bounds access
  · intro reverse executed
    obtain ⟨rev,run,revInv⟩ := KernelReverseLoop.executes M s f scale bh iv u 0 _ inv
    have stepEq : FP.Structural.step (engine M) (KernelProgram.loop .backward s f scale)
        (KernelPrefix.backward M s f scale u).val=some rev := by
      simpa [KernelProgram.loop,FP.Structural.step,evalOp_unfold] using run
    have same : reverse=rev := by simpa only [stepEq,Option.some.injEq] using executed.symm
    subst reverse
    rw [backward_tail,List.append_assoc,Safety.traceSafeList_append]
    refine ⟨middle_safe M s f scale bh iv _ rev revInv.carry revInv.static bounds access,?_⟩
    intro middle ran
    let mid := KernelBackwardMiddle.execute M s f scale bh iv
      (KernelReverseMemory.carry M s f scale bh iv u s.T) rev revInv.carry revInv.static
    have same : middle=mid.val := by simpa only [mid.property,Option.some.injEq] using ran.symm
    subst middle
    have middleMemory : mid.val.mem=(KernelBackward.start M s f scale bh iv u).mem := by
      change (KernelBackwardMiddle.execute M s f scale bh iv _ rev revInv.carry revInv.static).val.mem=_
      rw [KernelBackwardMiddle.memory]
      unfold KernelBackward.start
      cases hf : f.initialGrad <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
      · exact revInv.memory
      · exact KernelBackwardStateMemory.mem_congr M s "dh0" bh iv _ _ _ revInv.memory
    have seed : (KernelBackwardReference.input M s f bh iv rev).initial=
        (KernelBackwardReference.input M s f bh iv (KernelBackward.start M s f scale bh iv u)).initial := by
      simp only [KernelBackwardReference.input,KernelReference.read,revInv.memory]
      unfold KernelBackward.start
      cases hf : f.initialGrad <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
      simp only [KernelBackwardStateMemory.frame M s "dh0" bh iv _ _ "h0" _ (Or.inl (by decide))]
    have middleRegs := KernelBackwardMiddle.registers M s f scale bh iv
      (KernelReverseMemory.carry M s f scale bh iv u s.T) rev revInv.carry revInv.static
    rw [seed] at middleRegs
    have next : KernelRecomputeLoop.Holds M s f scale bh iv
        (KernelBackward.start M s f scale bh iv u) 0 mid.val :=
      ⟨Nat.zero_le _,middleRegs,middleMemory⟩
    have finalNil : (second s f scale).after=[] := by
      cases hb : f.vectorBeta <;>
        simp only [second,first,source,Source.backward,hb,Bool.false_eq_true,ite_false,ite_true,
          ComputeKernel.surfaceBody,VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,
          String.reduceEq,beq_self_eq_true,↓reduceIte] <;> rfl
    simp only [List.singleton_append,Safety.traceSafeList_cons,finalNil,Safety.traceSafeList_nil,
      implies_true,and_true]
    simpa only [recompute,Safety.stepSafe_unfold] using
      recompute_loop_safe M s f scale bh iv _ 0 _ next bounds access

#axiomsClean reverse_loop_safe
#axiomsClean recompute_loop_safe
#axiomsClean whole
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.SourceSafetyBackward
