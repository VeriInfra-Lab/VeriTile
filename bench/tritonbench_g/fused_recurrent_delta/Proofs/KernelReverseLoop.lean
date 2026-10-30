import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelReverseMemory
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardInitial
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReverseLoop
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters KernelBackwardRegisters
open KernelReverseMemory (carry)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 18000000
variable {α : Type} [Inhabited α]

structure Holds (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) : Prop where
  bound : n≤s.T
  regs : n<s.T → Reverse s f bh iv (s.T-1-n) (carry M s f scale bh iv initial n) u
  carry : u.regs .real [s.BK,s.BBV] "d_h"=some (KernelReverseMemory.carry M s f scale bh iv initial n)
  static : KernelBackwardInitial.Static s bh iv u
  memory : u.mem=(KernelReverseMemory.memory M s f scale bh iv initial n).mem

theorem initial (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α)
    (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    Holds M s f scale bh iv u 0 (KernelPrefix.backward M s f scale u).val := by
  have regs := KernelBackwardInitial.registers M s f scale bh iv u hv hk hh
  exact ⟨Nat.zero_le _,fun _ => regs,regs.carry,
    KernelBackwardInitial.static M s f scale bh iv u hv hk hh,KernelInitial.backward_memory M s f scale u⟩

theorem iteration (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) (inside : n<s.T)
    (inv : Holds M s f scale bh iv initial n u) :
    ∃ v,run (engine M) (first .backward s f scale).body u=some v ∧
      Holds M s f scale bh iv initial (n+1) v := by
  let h := carry M s f scale bh iv initial n
  let regs := inv.regs inside
  let result := KernelBackwardStep.reverse M s f scale bh iv (s.T-1-n) h u regs
  obtain ⟨next,dk,dv,dbeta⟩ := KernelReverseMemory.step_values M s f scale bh iv initial u n inv.memory
  refine ⟨result.val,result.property,⟨by omega,?_,?_,?_,?_⟩⟩
  · intro active
    have time : s.T-1-n=(s.T-1-(n+1))+1 := by omega
    have oldRegs : Reverse s f bh iv ((s.T-1-(n+1))+1) h u := by simpa only [←time] using regs
    have newRegs := KernelBackwardValues.registers M s f scale bh iv (s.T-1-(n+1)) h u oldRegs
    simpa only [←time,h,next] using newRegs
  · simpa only [h,next] using KernelBackwardValues.carry M s f scale bh iv (s.T-1-n) h u regs
  · exact KernelBackwardInitial.reverse_static M s f scale bh iv (s.T-1-n) h u regs inv.static
  · change (KernelBackwardStep.reverse M s f scale bh iv (s.T-1-n) h u regs).val.mem=_
    rw [KernelBackwardMemory.memory]
    dsimp only [h] at *
    rw [dk,dv,dbeta]
    unfold KernelReverseMemory.memory KernelBackwardMemory.betaStore
    split
    all_goals
      unfold KernelMemory.rowStore KernelBackwardMemory.keyStore
      first | apply KernelStore.masked_mem | apply KernelBackwardMemory.write_mem
      apply KernelStore.masked_mem
      apply KernelStore.masked_mem
      exact inv.memory

theorem bind_holds (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M s f scale bh iv initial n u) :
    Holds M s f scale bh iv initial n (u.setReg "_i" .nat [] (fun _ => n)) := by
  refine ⟨inv.bound,?_,?_,?_,inv.memory⟩
  · intro active
    rcases inv.regs active with ⟨hq,hk,hv,ho,hdk,hdv,hb,hdb,hmk,hmv,hh⟩
    refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
    all_goals simp_all [State.setReg]
  · simpa [State.setReg] using inv.carry
  · rcases inv.static with ⟨hh,hk,hv,hmk,hmv⟩
    refine ⟨?_,?_,?_,?_,?_⟩
    all_goals simp_all [State.setReg]

theorem executes (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M s f scale bh iv initial n u) :
    ∃ v,FP.Structural.range (engine M) "_i" n s.T 1 (first .backward s f scale).body u=some v ∧
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
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReverseLoop
