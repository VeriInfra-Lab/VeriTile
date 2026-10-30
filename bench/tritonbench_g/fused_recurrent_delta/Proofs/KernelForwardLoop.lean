import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardMemory
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardReferenceMemory
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelStatic
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardLoop
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
variable {α : Type} [Inhabited α]

def carry (M : Algebra α) (s : Shape) (f : Flags) (bh iv : Nat) (initial : State α) (n : Nat) :=
  KernelReference.forward M s (KernelReference.forwardInput M s f bh iv initial) n
structure Holds (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) : Prop where
  bound : n≤s.T
  regs : KernelRegisters.Forward s f bh iv n (carry M s f bh iv initial n) u
  static : KernelStatic.Forward s bh iv u
  memory : u.mem=(KernelForwardReferenceMemory.memory M s f scale bh iv initial n).mem

theorem initial (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α)
    (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    Holds M s f scale bh iv u 0 (KernelPrefix.forward M s f scale u).val :=
  ⟨Nat.zero_le _,KernelInitial.forward_registers M s f scale bh iv u hv hk hh,
    KernelStatic.initial M s f scale bh iv u hv hk hh,KernelInitial.forward_memory M s f scale u⟩

theorem iteration (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) (inside : n<s.T)
    (inv : Holds M s f scale bh iv initial n u) :
    ∃ v,run (engine M) (first .forward s f scale).body u=some v ∧
      Holds M s f scale bh iv initial (n+1) v := by
  let h := carry M s f bh iv initial n
  let result := KernelForwardStep.execute M s f scale bh iv n h u inv.regs
  obtain ⟨next,residual,output⟩ := KernelForwardReferenceMemory.step_values M s f scale bh iv initial u n inv.memory
  refine ⟨result.val,result.property,⟨by omega,?_,?_,?_⟩⟩
  · simpa only [h,carry,next] using KernelForwardValues.registers M s f scale bh iv n h u inv.regs
  · exact KernelStatic.step M s f scale bh iv n h u inv.regs inv.static
  · change (KernelForwardStep.execute M s f scale bh iv n h u inv.regs).val.mem=_
    rw [KernelForwardMemory.memory]
    dsimp only [h,carry] at *
    rw [residual,output]
    unfold KernelForwardReferenceMemory.memory KernelMemory.rowStore
    apply KernelStore.masked_mem
    apply KernelStore.masked_mem
    exact inv.memory

theorem bind_holds (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M s f scale bh iv initial n u) :
    Holds M s f scale bh iv initial n (u.setReg "_i" .nat [] (fun _ => n)) := by
  refine ⟨inv.bound,?_,?_,inv.memory⟩
  · rcases inv.regs with ⟨hq,hk,hv,ho,hb,hmk,hmv,hh⟩
    refine ⟨?_,?_,?_,?_,?_,?_,?_,?_⟩
    all_goals simp_all [State.setReg]
  · rcases inv.static with ⟨hh,hk,hv,hm⟩
    refine ⟨?_,?_,?_,?_⟩
    all_goals simp_all [State.setReg]

theorem executes (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M s f scale bh iv initial n u) :
    ∃ v,FP.Structural.range (engine M) "_i" n s.T 1 (first .forward s f scale).body u=some v ∧
      Holds M s f scale bh iv initial s.T v := by
  by_cases inside : n<s.T
  · obtain ⟨v,he,hv⟩ := iteration M s f scale bh iv initial n
      (u.setReg "_i" .nat [] (fun _ => n)) inside (bind_holds M s f scale bh iv initial n u inv)
    obtain ⟨w,hw,hf⟩ := executes M s f scale bh iv initial (n+1) v hv
    refine ⟨w,?_,hf⟩
    rw [FP.Structural.range]
    simpa [inside,he] using hw
  · have bound := inv.bound
    have equal : n=s.T := by omega
    exact ⟨u,FP.Structural.range_done _ _ _ _ _ _ _ (by omega),by simpa [equal] using inv⟩
termination_by s.T-n

#axiomsClean initial
#axiomsClean iteration
#axiomsClean executes
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardLoop
