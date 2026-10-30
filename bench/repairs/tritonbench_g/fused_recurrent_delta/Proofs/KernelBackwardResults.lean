import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelReverseMemory
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelRecomputeReferenceMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardResults
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters KernelBackwardMemory
open KernelRecomputeReferenceMemory (key_other_time)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 18000000
variable {α : Type} [Inhabited α]

theorem reverse_key (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (count n : Nat) (inside : n<count) (bound : count≤s.T)
    (i : Fin s.BK) (active : i.val<s.K) :
    (KernelReverseMemory.memory M s f scale bh iv initial count).mem "dk"
      (keyAddress s (plane s bh iv) (s.T-1-n) i.val)=KernelReference.stored M
        (KernelBackwardReference.dk M s (KernelBackwardReference.input M s f bh iv initial)
          (s.T-1-n) (KernelReverseMemory.enhanced M s f scale bh iv initial n) i) := by
  induction count with
  | zero => omega
  | succ count ih =>
    unfold KernelReverseMemory.memory
    rw [beta_frame M s f _ _ _ _ _ "dk" _ (by decide),
      KernelMemory.rowStore_frame M s "dv" _ _ _ _ _ _ "dk" _ (Or.inl (by decide))]
    by_cases current : n=count
    · subst n
      exact key_cell M s "dk" (plane s bh iv) _ _ _ i active
    · rw [key_other_time M s "dk" (plane s bh iv) (s.T-1-count) (s.T-1-n) _ _ i active (by omega)]
      exact ih (by omega) (by omega)

theorem reverse_value (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (count n : Nat) (inside : n<count) (bound : count≤s.T)
    (j : Fin s.BBV) (active : iv*s.BBV+j.val<s.V) :
    (KernelReverseMemory.memory M s f scale bh iv initial count).mem "dv"
      (valAddress s bh iv s.BBV (s.T-1-n) j.val)=KernelReference.stored M
        (KernelBackwardReference.dv M s (KernelBackwardReference.input M s f bh iv initial)
          (s.T-1-n) (KernelReverseMemory.enhanced M s f scale bh iv initial n) j) := by
  induction count with
  | zero => omega
  | succ count ih =>
    unfold KernelReverseMemory.memory
    rw [beta_frame M s f _ _ _ _ _ "dv" _ (by decide)]
    by_cases current : n=count
    · subst n
      exact KernelMemory.rowStore_cell M s "dv" bh iv s.BBV _ _ _ j active
    · rw [KernelMemory.rowStore_other_time M s "dv" bh iv s.BBV (s.T-1-count) (s.T-1-n) _ _ j active (by omega),
        key_frame M s "dk" _ _ _ _ "dv" _ (Or.inl (by decide))]
      exact ih (by omega) (by omega)

theorem reverse_vector_beta (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (count n : Nat) (inside : n<count) (bound : count≤s.T) (vector : f.vectorBeta=Bool.true)
    (j : Fin s.BBV) (active : iv*s.BBV+j.val<s.V) :
    (KernelReverseMemory.memory M s f scale bh iv initial count).mem "dbeta"
      (valAddress s (plane s bh iv) iv s.BBV (s.T-1-n) j.val)=KernelReference.stored M
        (KernelBackwardReference.dbeta M s (KernelBackwardReference.input M s f bh iv initial)
          (s.T-1-n) (KernelReverseMemory.enhanced M s f scale bh iv initial n) j) := by
  induction count with
  | zero => omega
  | succ count ih =>
    unfold KernelReverseMemory.memory
    simp only [betaStore,vector,ite_true]
    by_cases current : n=count
    · subst n
      exact KernelMemory.rowStore_cell M s "dbeta" (plane s bh iv) iv s.BBV _ _ _ j active
    · rw [KernelMemory.rowStore_other_time M s "dbeta" (plane s bh iv) iv s.BBV (s.T-1-count) (s.T-1-n) _ _ j active (by omega),
        KernelMemory.rowStore_frame M s "dv" _ _ _ _ _ _ "dbeta" _ (Or.inl (by decide)),
        key_frame M s "dk" _ _ _ _ "dbeta" _ (Or.inl (by decide))]
      exact ih (by omega) (by omega)

theorem reverse_scalar_beta (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (count n : Nat) (inside : n<count) (bound : count≤s.T) (vector : f.vectorBeta=Bool.false) :
    (KernelReverseMemory.memory M s f scale bh iv initial count).mem "dbeta"
      (plane s bh iv*s.T+(s.T-1-n))=KernelReference.stored M
        (KernelBackwardReference.betaSum M s (KernelBackwardReference.dbeta M s
          (KernelBackwardReference.input M s f bh iv initial)
          (s.T-1-n) (KernelReverseMemory.enhanced M s f scale bh iv initial n))) := by
  induction count with
  | zero => omega
  | succ count ih =>
    unfold KernelReverseMemory.memory
    simp only [betaStore,vector,Bool.false_eq_true,ite_false]
    by_cases current : n=count
    · subst n
      exact State.write_same _ _ _ _
    · rw [State.write_other _ _ _ _ _ _ (Or.inr (by omega)),
        KernelMemory.rowStore_frame M s "dv" _ _ _ _ _ _ "dbeta" _ (Or.inl (by decide)),
        key_frame M s "dk" _ _ _ _ "dbeta" _ (Or.inl (by decide))]
      exact ih (by omega) (by omega)

theorem recompute_query (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n t : Nat) (inside : t<n) (i : Fin s.BK) (active : i.val<s.K) :
    (KernelRecomputeReferenceMemory.memory M s f scale bh iv initial n).mem "dq"
      (keyAddress s (plane s bh iv) t i.val)=
        KernelReference.stored M (KernelRecomputeReferenceMemory.query M s f scale bh iv initial t i) := by
  induction n with
  | zero => omega
  | succ n ih =>
    unfold KernelRecomputeReferenceMemory.memory
    by_cases current : t=n
    · subst t
      exact key_cell M s "dq" (plane s bh iv) n _ _ i active
    · rw [key_other_time M s "dq" (plane s bh iv) n t _ _ i active current,
        key_frame M s "dk" _ _ _ _ "dq" _ (Or.inl (by decide))]
      exact ih (by omega)

theorem recompute_key (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n t : Nat) (inside : t<n) (i : Fin s.BK) (active : i.val<s.K) :
    (KernelRecomputeReferenceMemory.memory M s f scale bh iv initial n).mem "dk"
      (keyAddress s (plane s bh iv) t i.val)=
        KernelReference.stored M (KernelRecomputeReferenceMemory.corrected M s f bh iv initial t i) := by
  induction n with
  | zero => omega
  | succ n ih =>
    unfold KernelRecomputeReferenceMemory.memory
    rw [key_frame M s "dq" _ _ _ _ "dk" _ (Or.inl (by decide))]
    by_cases current : t=n
    · subst t
      exact key_cell M s "dk" (plane s bh iv) n _ _ i active
    · rw [key_other_time M s "dk" (plane s bh iv) n t _ _ i active current]
      exact ih (by omega)

#axiomsClean reverse_key
#axiomsClean reverse_value
#axiomsClean reverse_vector_beta
#axiomsClean reverse_scalar_beta
#axiomsClean recompute_query
#axiomsClean recompute_key
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardResults
