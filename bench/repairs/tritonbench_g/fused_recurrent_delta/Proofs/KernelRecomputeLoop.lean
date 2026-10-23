import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelRecomputeReferenceMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRecomputeLoop
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters KernelBackwardRegisters
open KernelRecomputeReferenceMemory (carry)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
variable {α : Type} [Inhabited α]

structure Holds (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) : Prop where
  bound : n≤s.T
  regs : Recompute s f bh iv n (carry M s f bh iv initial n) u
  memory : u.mem=(KernelRecomputeReferenceMemory.memory M s f scale bh iv initial n).mem

theorem iteration (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) (inside : n<s.T)
    (inv : Holds M s f scale bh iv initial n u) :
    ∃ v,run (engine M) (second s f scale).body u=some v ∧
      Holds M s f scale bh iv initial (n+1) v := by
  let h := carry M s f bh iv initial n
  let result := KernelBackwardStep.recompute M s f scale bh iv n h u inv.regs
  obtain ⟨next,dk,dq⟩ := KernelRecomputeReferenceMemory.step_values M s f scale bh iv initial u n inv.memory
  refine ⟨result.val,result.property,⟨by omega,?_,?_⟩⟩
  · simpa only [h,next] using KernelRecomputeValues.registers M s f scale bh iv n h u inv.regs
  · change (KernelBackwardStep.recompute M s f scale bh iv n h u inv.regs).val.mem=_
    rw [KernelRecomputeMemory.memory]
    dsimp only [h] at *
    rw [dk,dq]
    unfold KernelRecomputeReferenceMemory.memory KernelBackwardMemory.keyStore
    apply KernelStore.masked_mem
    apply KernelStore.masked_mem
    exact inv.memory

theorem bind_holds (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M s f scale bh iv initial n u) :
    Holds M s f scale bh iv initial n (u.setReg "i" .nat [] (fun _ => n)) := by
  refine ⟨inv.bound,?_,inv.memory⟩
  rcases inv.regs with ⟨hk,hv,ho,hdq,hdk,hdv,hb,hmk,hmv,hh⟩
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  all_goals simp_all [State.setReg]

theorem executes (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M s f scale bh iv initial n u) :
    ∃ v,FP.Structural.range (engine M) "i" n s.T 1 (second s f scale).body u=some v ∧
      Holds M s f scale bh iv initial s.T v := by
  by_cases inside : n<s.T
  · obtain ⟨v,he,hv⟩ := iteration M s f scale bh iv initial n
      (u.setReg "i" .nat [] (fun _ => n)) inside (bind_holds M s f scale bh iv initial n u inv)
    obtain ⟨w,hw,hf⟩ := executes M s f scale bh iv initial (n+1) v hv
    refine ⟨w,?_,hf⟩
    rw [FP.Structural.range]
    simpa [inside,he] using hw
  · have bound := inv.bound
    have equal : n=s.T := by omega
    exact ⟨u,FP.Structural.range_done _ _ _ _ _ _ _ (by omega),by simpa [equal] using inv⟩
termination_by s.T-n

#axiomsClean iteration
#axiomsClean executes
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRecomputeLoop
