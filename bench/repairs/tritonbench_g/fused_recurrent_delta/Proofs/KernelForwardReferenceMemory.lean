import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardValues
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardReferenceMemory
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelMemory
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
variable {α : Type} [Inhabited α]

def memory (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (initial : State α) : Nat → State α
  | 0 => initial
  | n+1 =>
      let a := KernelReference.forwardInput M s f bh iv initial
      KernelMemory.rowStore M s "o" bh iv s.FBV n (KernelReference.output M s a scale n)
        (KernelMemory.rowStore M s "v" bh iv s.FBV n (KernelReference.residual M s a n)
          (memory M s f scale bh iv initial n))

theorem readonly (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (initial : State α)
    (n : Nat) (r : RegionName) (off : Nat) (ho : r≠"o") (hv : r≠"v") :
    (memory M s f scale bh iv initial n).mem r off=initial.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    unfold memory
    rw [rowStore_frame M s "o" _ _ _ _ _ _ r off (Or.inl ho),
      rowStore_frame M s "v" _ _ _ _ _ _ r off (Or.inl hv),ih]

theorem future (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (initial : State α)
    (n t : Nat) (future : n≤t) (j : Fin s.FBV) (active : iv*s.FBV+j.val<s.V) :
    (memory M s f scale bh iv initial n).mem "v" (valAddress s bh iv s.FBV t j.val)=
      initial.mem "v" (valAddress s bh iv s.FBV t j.val) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    unfold memory
    rw [rowStore_frame M s "o" _ _ _ _ _ _ "v" _ (Or.inl (by decide)),
      rowStore_other_time M s "v" bh iv s.FBV n t _ _ j active (by omega)]
    exact ih (by omega)

theorem key_current (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial u : State α) (n t : Nat) (same : u.mem=(memory M s f scale bh iv initial n).mem) :
    (KernelReference.forwardInput M s f bh iv u).key t=
      (KernelReference.forwardInput M s f bh iv initial).key t := by
  funext i
  simp only [KernelReference.forwardInput,KernelReference.read,same,
    readonly M s f scale bh iv initial n "k" _ (by decide) (by decide)]
theorem query_current (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial u : State α) (n t : Nat) (same : u.mem=(memory M s f scale bh iv initial n).mem) :
    (KernelReference.forwardInput M s f bh iv u).query t=
      (KernelReference.forwardInput M s f bh iv initial).query t := by
  funext i
  simp only [KernelReference.forwardInput,KernelReference.read,same,
    readonly M s f scale bh iv initial n "q" _ (by decide) (by decide)]
theorem beta_current (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial u : State α) (n t : Nat) (same : u.mem=(memory M s f scale bh iv initial n).mem) :
    (KernelReference.forwardInput M s f bh iv u).beta t=
      (KernelReference.forwardInput M s f bh iv initial).beta t := by
  funext j
  simp only [KernelReference.forwardInput,KernelReference.read,same,
    readonly M s f scale bh iv initial n "beta" _ (by decide) (by decide)]
theorem value_current (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial u : State α) (n t : Nat) (futureTime : n≤t)
    (same : u.mem=(memory M s f scale bh iv initial n).mem) :
    (KernelReference.forwardInput M s f bh iv u).value t=
      (KernelReference.forwardInput M s f bh iv initial).value t := by
  funext j
  simp only [KernelReference.forwardInput,KernelReference.read,same]
  by_cases active : iv*s.FBV+j.val<s.V
  · rw [future M s f scale bh iv initial n t futureTime j active]
  · simp [active]

theorem step_values (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial u : State α) (n : Nat) (same : u.mem=(memory M s f scale bh iv initial n).mem) :
    let a := KernelReference.forwardInput M s f bh iv initial
    let h := KernelReference.forward M s a n
    KernelForwardValues.next M s f bh iv n h u=KernelReference.forward M s a (n+1) ∧
    KernelForwardValues.residual M s f bh iv n h u=KernelReference.residual M s a n ∧
    KernelForwardValues.output M s f scale bh iv n h u=KernelReference.output M s a scale n := by
  have hk := key_current M s f scale bh iv initial u n n same
  have hq := query_current M s f scale bh iv initial u n n same
  have hb := beta_current M s f scale bh iv initial u n n same
  have hv := value_current M s f scale bh iv initial u n n (Nat.le_refl _) same
  dsimp only
  refine ⟨?_,?_,?_⟩
  all_goals funext i
  all_goals simp only [KernelForwardValues.next,KernelForwardValues.residual,KernelForwardValues.output,
    KernelForwardValues.input,KernelReference.output,KernelReference.residual,KernelReference.forward,
    Nat.add_zero,hk,hq,hb,hv]

#axiomsClean readonly
#axiomsClean future
#axiomsClean step_values
#specNonCircular memory avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardReferenceMemory
