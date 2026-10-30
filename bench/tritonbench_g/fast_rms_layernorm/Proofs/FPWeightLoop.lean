import bench.tritonbench_g.fast_rms_layernorm.Proofs.FPWeightStep

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPWeightLoop
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPWeightProgram FPWeightStep
noncomputable section
set_option maxRecDepth 25000
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

def carry (M : Algebra α) (h : Call) (K pid : Nat) (initial : State α) (n : Nat) : Values α .real [K] :=
  fun i => FPWeightReference.carry M h.gemma
    (fun row => raw M h K pid initial "x" row i)
    (fun row => raw M h K pid initial "dx" row i) (saved initial) n

structure Holds (M : Algebra α) (h : Call) (K pid : Nat) (initial : State α) (n : Nat) (u : State α) : Prop where
  bound : n≤h.rows
  col : u.regs .nat [K] "col"=some (columns K pid)
  mask : u.regs .bool [K] "mask"=some (masks h K pid)
  acc : u.regs .real [K] "acc"=some (carry M h K pid initial n)
  mem : u.mem=initial.mem

theorem initial_holds (M : Algebra α) (h : Call) (K : Nat) (u : State α) :
    Holds M h K (u.pids 0) u 0 (FPWeightStep.initial M h K u) := by
  refine ⟨Nat.zero_le _,?_,?_,?_,rfl⟩
  · simp [FPWeightStep.initial]
  · simp [FPWeightStep.initial]
  · simp [FPWeightStep.initial,carry,FPWeightReference.carry]
    rfl

theorem loop_executes (M : Algebra α) (h : Call) (K pid : Nat) (initial : State α)
    (n : Nat) (u : State α) (inv : Holds M h K pid initial n u) :
    ∃ v, FP.Structural.range (engine M) "row" n h.rows 1 (loopBody h K) u=some v ∧
      Holds M h K pid initial h.rows v := by
  by_cases hn : n<h.rows
  · obtain ⟨v,he,hmem,_,hcol,hmask,hacc⟩ := iteration M h K pid n u
      (carry M h K pid initial n) inv.col inv.mask inv.acc
    have next : nextAcc M h K pid n u (carry M h K pid initial n)=carry M h K pid initial (n+1) := by
      funext i
      simp [nextAcc,carry,FPWeightReference.carry,raw,saved,inv.mem]
    have hv : Holds M h K pid initial (n+1) v :=
      ⟨Nat.succ_le_of_lt hn,hcol,hmask,by rw [next] at hacc; exact hacc,hmem.trans inv.mem⟩
    obtain ⟨w,hw,hf⟩ := loop_executes M h K pid initial (n+1) v hv
    refine ⟨w,?_,hf⟩
    rw [FP.Structural.range]
    simpa [hn,he] using hw
  · have eq : n=h.rows := Nat.le_antisymm inv.bound (Nat.le_of_not_gt hn)
    refine ⟨u,?_,by simpa [eq] using inv⟩
    rw [FP.Structural.range]
    simp [hn]
termination_by h.rows-n

#axiomsClean loop_executes
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPWeightLoop
