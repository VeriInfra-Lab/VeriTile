import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackward
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardFrame
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardFacts
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters KernelBackwardRegisters KernelBackward
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000
variable {α : Type} [Inhabited α]

def enhancedAt (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α) (t : Nat) :=
  KernelReverseMemory.enhanced M s f scale bh iv u (s.T-1-t)
def value (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α) (t : Nat) :=
  KernelBackwardReference.dv M s (KernelBackwardReference.input M s f bh iv u) t
    (enhancedAt M s f scale bh iv u t)
def beta (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α) (t : Nat) :=
  KernelBackwardReference.dbeta M s (KernelBackwardReference.input M s f bh iv u) t
    (enhancedAt M s f scale bh iv u t)

theorem start_other (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α)
    (r : RegionName) (off : Nat) (different : r≠"dh0") :
    (start M s f scale bh iv u).mem r off=(KernelReverseMemory.memory M s f scale bh iv u s.T).mem r off := by
  unfold start
  split
  · exact KernelBackwardStateMemory.frame M s "dh0" _ _ _ _ r off (Or.inl different)
  · rfl

structure Facts (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u v : State α) : Prop where
  success : FP.Structural.exec (engine M) (source .backward s f scale) u=some v
  query : ∀ (t : Fin s.T) (i : Fin s.BK),i.val<s.K →
    v.mem "dq" (keyAddress s (plane s bh iv) t.val i.val)=KernelReference.stored M
      (KernelRecomputeReferenceMemory.query M s f scale bh iv (start M s f scale bh iv u) t.val i)
  key : ∀ (t : Fin s.T) (i : Fin s.BK),i.val<s.K →
    v.mem "dk" (keyAddress s (plane s bh iv) t.val i.val)=KernelReference.stored M
      (KernelRecomputeReferenceMemory.corrected M s f bh iv (start M s f scale bh iv u) t.val i)
  value : ∀ (t : Fin s.T) (j : Fin s.BBV),iv*s.BBV+j.val<s.V →
    v.mem "dv" (valAddress s bh iv s.BBV t.val j.val)=KernelReference.stored M
      (KernelBackwardFacts.value M s f scale bh iv u t.val j)
  vectorBeta : f.vectorBeta=Bool.true → ∀ (t : Fin s.T) (j : Fin s.BBV),iv*s.BBV+j.val<s.V →
    v.mem "dbeta" (valAddress s (plane s bh iv) iv s.BBV t.val j.val)=KernelReference.stored M
      (beta M s f scale bh iv u t.val j)
  scalarBeta : f.vectorBeta=Bool.false → ∀ t : Fin s.T,
    v.mem "dbeta" (plane s bh iv*s.T+t.val)=KernelReference.stored M
      (KernelBackwardReference.betaSum M s (beta M s f scale bh iv u t.val))
  seed : f.initialGrad=Bool.true → ∀ (i : Fin s.BK) (j : Fin s.BBV),i.val<s.K → iv*s.BBV+j.val<s.V →
    v.mem "dh0" (stateAddress s bh iv s.BBV i.val j.val)=KernelReference.stored M
      (KernelReverseMemory.carry M s f scale bh iv u s.T (i,j,PUnit.unit))
  frame : ∀ r off,¬KernelBackwardFrame.Writes s f bh iv r off → v.mem r off=u.mem r off

theorem complete (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α)
    (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    ∃ v,Facts M s f scale bh iv u v := by
  obtain ⟨v,success,memory⟩ := KernelBackward.executes M s f scale bh iv u hv hk hh
  refine ⟨v,success,?_,?_,?_,?_,?_,?_,?_⟩
  · intro t i active
    rw [memory]
    exact KernelBackwardResults.recompute_query M s f scale bh iv _ s.T t.val t.isLt i active
  · intro t i active
    rw [memory]
    exact KernelBackwardResults.recompute_key M s f scale bh iv _ s.T t.val t.isLt i active
  · intro t j active
    rw [memory]
    unfold result
    rw [KernelRecomputeReferenceMemory.readonly M s f scale bh iv _ s.T "dv" _ (by decide) (by decide),
      start_other M s f scale bh iv u "dv" _ (by decide)]
    have time : s.T-1-(s.T-1-t.val)=t.val := by omega
    simpa only [time,value,enhancedAt] using
      KernelBackwardResults.reverse_value M s f scale bh iv u s.T (s.T-1-t.val) (by omega) (Nat.le_refl _) j active
  · intro vector t j active
    rw [memory]
    unfold result
    rw [KernelRecomputeReferenceMemory.readonly M s f scale bh iv _ s.T "dbeta" _ (by decide) (by decide),
      start_other M s f scale bh iv u "dbeta" _ (by decide)]
    have time : s.T-1-(s.T-1-t.val)=t.val := by omega
    simpa only [time,beta,enhancedAt] using
      KernelBackwardResults.reverse_vector_beta M s f scale bh iv u s.T (s.T-1-t.val) (by omega) (Nat.le_refl _) vector j active
  · intro scalar t
    rw [memory]
    unfold result
    rw [KernelRecomputeReferenceMemory.readonly M s f scale bh iv _ s.T "dbeta" _ (by decide) (by decide),
      start_other M s f scale bh iv u "dbeta" _ (by decide)]
    have time : s.T-1-(s.T-1-t.val)=t.val := by omega
    simpa only [time,beta,enhancedAt] using
      KernelBackwardResults.reverse_scalar_beta M s f scale bh iv u s.T (s.T-1-t.val) (by omega) (Nat.le_refl _) scalar
  · intro seed i j hi hj
    rw [memory]
    unfold result
    rw [KernelRecomputeReferenceMemory.readonly M s f scale bh iv _ s.T "dh0" _ (by decide) (by decide)]
    simp only [start,seed,ite_true]
    exact KernelBackwardStateMemory.cell M s "dh0" bh iv _ _ i j hi hj
  · intro r off outside
    rw [memory]
    unfold result
    rw [KernelBackwardFrame.recompute_frame M s f scale bh iv _ s.T (Nat.le_refl _) r off outside]
    unfold start
    split
    · rename_i seed
      rw [KernelBackwardStateMemory.frame]
      · exact KernelBackwardFrame.reverse_frame M s f scale bh iv u s.T (Nat.le_refl _) r off outside
      · by_cases different : r≠"dh0"
        · exact Or.inl different
        · right
          intro i j hi hj same
          exact outside (Or.inr (Or.inr (Or.inr ⟨seed,not_ne_iff.mp different,i,j,hi,hj,same⟩)))
    · exact KernelBackwardFrame.reverse_frame M s f scale bh iv u s.T (Nat.le_refl _) r off outside

#axiomsClean complete
#specNonCircular beta avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardFacts
