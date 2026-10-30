import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightStep

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightLoop
open VeriTile.Triton RealWeightProgram RealWeightStep
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000

def carry (h : Call) (K pid : Nat) (initial : BlockState) (n : Nat) (j : Fin K) : ℝ :=
  ∑ row∈Finset.range n,raw h K pid initial "dx" row j*(raw h K pid initial "x" row j*saved initial row)

structure Holds (h : Call) (K pid : Nat) (initial : BlockState) (n : Nat) (u : BlockState) : Prop where
  bound : n≤h.rows
  col : u.regs .int [K] "col"=some (columns K pid)
  mask : u.regs .bool [K] "mask"=some (masks h K pid)
  acc : u.regs .real [K] "acc"=some (Tile.vec fun j => some (carry h K pid initial n j))
  mem : u.mem=initial.mem

theorem initial_holds (h : Call) (K : Nat) (u : BlockState) :
    Holds h K (u.pids 0) u 0 (RealWeightStep.initial h K u) := by
  refine ⟨Nat.zero_le _,?_,?_,?_,rfl⟩
  · simp [RealWeightStep.initial]
  · simp [RealWeightStep.initial]
  · simp [RealWeightStep.initial,carry]

theorem loop_executes (h : Call) (K pid : Nat) (initial : BlockState)
    (n : Nat) (u : BlockState) (inv : Holds h K pid initial n u) :
    ∃ v,stepForRangeIntAux "row" (n : Int) (h.rows : Int) 1 (loopBody h K) u=some v ∧ Holds h K pid initial h.rows v := by
  by_cases hn : n<h.rows
  · obtain ⟨v,he,hmem,_,hcol,hmask,hacc⟩ := iteration h K pid n u
      (carry h K pid initial n) inv.col inv.mask inv.acc
    have next : nextAcc h K pid n u (carry h K pid initial n)=carry h K pid initial (n+1) := by
      funext j
      by_cases active : pid*K+j.val<h.cols <;>
        simp [nextAcc,carry,Finset.sum_range_succ,raw,saved,BlockState.readMem,inv.mem,active]
    have hv : Holds h K pid initial (n+1) v :=
      ⟨Nat.succ_le_of_lt hn,hcol,hmask,by rw [next] at hacc; exact hacc,hmem.trans inv.mem⟩
    obtain ⟨w,hw,hf⟩ := loop_executes h K pid initial (n+1) v hv
    refine ⟨w,?_,hf⟩
    rw [stepForRangeIntAux]
    simpa [hn,he,Int.natCast_add] using hw
  · have eq : n=h.rows := Nat.le_antisymm inv.bound (Nat.le_of_not_gt hn)
    refine ⟨u,?_,by simpa [eq] using inv⟩
    rw [stepForRangeIntAux]
    simp [hn]
termination_by h.rows-n

#axiomsClean loop_executes
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightLoop
