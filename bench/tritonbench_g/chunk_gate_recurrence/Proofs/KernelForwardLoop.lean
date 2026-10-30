import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelStep
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReference
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelStore

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelForwardLoop
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelMemory
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def carry (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c)
    (u : State α) (n : Nat) : Lane → α :=
  KernelReference.forward M (KernelReference.read c s p u) seed n

structure Holds (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c)
    (initial : State α) (n : Nat) (u : State α) : Prop where
  bound : n≤c.N-1
  sourcePtr : u.regs .ptr [64,16] "S"=some (ptr "S" (fun e j => address c p n e.val j.val))
  targetPtr : u.regs .ptr [64,16] "O"=some (ptr "O" (fun e j => address c p (n+1) e.val j.val))
  gatePtr : u.regs .ptr [] "D"=some (fun _ : TileIndex [] => (("D" : RegionName),gateAddress c p n))
  acc : u.regs .real [64,16] "acc"=some (carry M c s seed p initial n)
  readsSource : ∀ off, u.mem "S" off=initial.mem "S" off
  readsGate : ∀ off, u.mem "D" off=initial.mem "D" off
  output : ∀ t, t≤n → ∀ (e : Fin 64) (j : Fin 16),
    u.mem "O" (address c p t e.val j.val)=
      KernelReference.stored M .real (carry M c s seed p initial t (e,j,PUnit.unit))
  frame : Frame .forward c p initial u

theorem iteration (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c)
    (initial : State α) (n : Nat) (u : State α) (hn : n<c.N-1)
    (inv : Holds M c s seed p initial n u) :
    ∃ v, run (engine M) (KernelOps.forwardBody c) u=some v ∧
      Holds M c s seed p initial (n+1) v := by
  let off := fun (e : Fin 64) (j : Fin 16) => address c p n e.val j.val
  let target := fun (e : Fin 64) (j : Fin 16) => address c p (n+1) e.val j.val
  let acc := carry M c s seed p initial n
  let value := KernelStep.forwardValue M off (gateAddress c p n) acc u
  let pre := KernelStep.forwardPre M off (gateAddress c p n) acc u
  let v := KernelStep.forwardResult M c off target (gateAddress c p n) acc u
  have values : value=carry M c s seed p initial (n+1) := by
    funext i
    simp [value,KernelStep.forwardValue,KernelStep.loadedMatrix,KernelStep.loadedScalar,
      KernelOps.next,carry,KernelReference.forward,KernelReference.update,KernelReference.read,
      raw,inv.readsSource,inv.readsGate,acc,off]
  have sourceNext : (fun e j => off e j+c.K*c.V)=target := by
    funext e j
    exact (address_next c p n e.val j.val).symm
  have targetNext : (fun e j => target e j+c.K*c.V)=
      (fun (e : Fin 64) (j : Fin 16) => address c p (n+1+1) e.val j.val) := by
    funext e j
    exact (address_next c p (n+1) e.val j.val).symm
  have gateNext : gateAddress c p n+1=gateAddress c p (n+1) := by simp [gateAddress,Nat.add_assoc]
  refine ⟨v,KernelStep.forward M c off target _ acc u inv.sourcePtr inv.targetPtr inv.gatePtr inv.acc,
    Nat.succ_le_of_lt hn,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [v,KernelStep.forwardResult,sourceNext,target]
  · simp [v,KernelStep.forwardResult,targetNext]
  · simp [v,KernelStep.forwardResult,gateNext]
  · simp [v,KernelStep.forwardResult,KernelStep.forwardPre]
    exact values
  · intro o
    change (scatter "O" .real target (fun i => M.cast (some .fp32) .real .real (value i)) pre).mem "S" o=_
    simpa [pre,KernelStep.forwardPre] using
      (scatter_other "O" .real target (fun i => M.cast (some .fp32) .real .real (value i)) pre "S" o (by decide)).trans (inv.readsSource o)
  · intro o
    change (scatter "O" .real target (fun i => M.cast (some .fp32) .real .real (value i)) pre).mem "D" o=_
    simpa [pre,KernelStep.forwardPre] using
      (scatter_other "O" .real target (fun i => M.cast (some .fp32) .real .real (value i)) pre "D" o (by decide)).trans (inv.readsGate o)
  · intro t ht e j
    change (scatter "O" .real target (fun i => M.cast (some .fp32) .real .real (value i)) pre).mem "O" _=_
    by_cases same : t=n+1
    · subst t
      rw [scatter_cell,values]
      rfl
    · rw [scatter_other_time c p "O" .real (n+1) t (Ne.symm same)]
      exact inv.output t (by omega) e j
  · intro r o outside
    change (scatter "O" .real target (fun i => M.cast (some .fp32) .real .real (value i)) pre).mem r o=_
    exact (forward_frame c p ⟨n+1,by omega⟩ _ pre r o outside).trans (inv.frame r o outside)

theorem bind_holds (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M c s seed p initial n u) :
    Holds M c s seed p initial n (u.setReg "_i" .nat [] (fun _ => n)) := by
  refine ⟨inv.bound,?_,?_,?_,?_,inv.readsSource,inv.readsGate,inv.output,inv.frame⟩
  all_goals simp [inv.sourcePtr,inv.targetPtr,inv.gatePtr,inv.acc]

theorem executes (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M c s seed p initial n u) :
    ∃ v, FP.Structural.range (engine M) "_i" n (c.N-1) 1 (KernelOps.forwardBody c) u=some v ∧
      Holds M c s seed p initial (c.N-1) v := by
  by_cases hn : n<c.N-1
  · obtain ⟨v,he,hv⟩ := iteration M c s seed p initial n
      (u.setReg "_i" .nat [] (fun _ => n)) hn (bind_holds M c s seed p initial n u inv)
    obtain ⟨w,hw,hf⟩ := executes M c s seed p initial (n+1) v hv
    refine ⟨w,?_,hf⟩
    rw [FP.Structural.range]
    simpa [hn,he] using hw
  · have bound := inv.bound
    have eq : n=c.N-1 := by omega
    refine ⟨u,FP.Structural.range_done _ _ _ _ _ _ _ (by omega),by simpa [eq] using inv⟩
termination_by c.N-1-n

#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelForwardLoop
