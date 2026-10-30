import bench.repairs.tritonbench_g.attention_fwd_triton2.Proofs.InputValues

/-! Reuse the established independent recurrence with only finite actual
input reads. No assumption is imposed on unrelated cells or inactive lanes. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.FiniteReference
open VeriTile Triton FP.Structural Whole001 Source Whole Machine InputValues
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem algorithm (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (canon : Canonical p) (hx : s.pids 0<(p.length+127)/128) (hy : s.pids 1<p.batches*p.heads)
    (read : RegionName→Nat→α)
    (loads : ∀ r off,ReadAddress p r off → (s.mem r off).read .real=read r off) :
    answer M dot p s=KernelOnly.Specification.algorithm M dot p read
      (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) := by
  have safe := Access.safe p canon
  have hz : s.pids 1/p.heads*p.heads+s.pids 1%p.heads=s.pids 1 := by
    simpa [Nat.mul_comm] using Nat.div_add_mod (s.pids 1) p.heads
  have q : Source.query M p s=KernelOnly.Specification.query M p read
      (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) := by
    funext i
    unfold Source.query KernelOnly.Specification.query
    by_cases active : s.pids 0*128+i.1.val<p.length ∧ i.2.1.val<96
    · rw [if_pos active,if_pos active]
      apply congrArg M.fp32Load
      apply loads
      exact Or.inl ⟨rfl,safe.query _ hx _ hy _ i.1.isLt _ i.2.1.isLt active.1 active.2⟩
    · rw [if_neg active,if_neg active]
      rfl
  have qs : Source.queryScale M p s=KernelOnly.Specification.queryScale M p read
      (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) := by
    simp only [Source.queryScale,KernelOnly.Specification.queryScale,hz]
    apply congrArg M.fp32Load
    apply loads
    exact Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl,safe.queryScale _ hx _ hy⟩)))
  have k : ∀ row,Source.key M p s row=KernelOnly.Specification.key M p read
      (s.pids 1/p.heads) (s.pids 1%p.heads) row := by
    intro row
    funext i
    unfold Source.key KernelOnly.Specification.key
    by_cases active : i.2.1.val<p.length-row ∧ i.1.val<96
    · rw [if_pos active,if_pos active]
      apply congrArg M.fp32Load
      have eq : base p s p.k+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row=
          offset p.k (s.pids 1/p.heads) (s.pids 1%p.heads) (row+i.2.1.val) i.1.val := by
        simp [base,offset,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
      rw [eq]
      apply loads
      exact Or.inr (Or.inl ⟨rfl,(safe.keyValue _ hy row (by omega) _ i.2.1.isLt _ i.1.isLt active.1 active.2).1⟩)
    · rw [if_neg active,if_neg active]
      rfl
  have ks : ∀ row,row<p.length → Source.keyScale M p s row=KernelOnly.Specification.keyScale M p read
      (s.pids 1/p.heads) (s.pids 1%p.heads) row := by
    intro row inside
    simp only [Source.keyScale,KernelOnly.Specification.keyScale,hz]
    apply congrArg M.fp32Load
    apply loads
    exact Or.inr (Or.inr (Or.inr (Or.inr ⟨rfl,safe.keyScale _ hy row inside⟩)))
  have v : ∀ row,Source.value M p s row=KernelOnly.Specification.value M p read
      (s.pids 1/p.heads) (s.pids 1%p.heads) row := by
    intro row
    funext i
    unfold Source.value KernelOnly.Specification.value
    by_cases active : i.1.val<p.length-row ∧ i.2.1.val<96
    · rw [if_pos active,if_pos active]
      apply congrArg (M.cast (some .fp32) .real .fp16)
      apply congrArg M.fp32Load
      apply loads
      exact Or.inr (Or.inr (Or.inl ⟨rfl,(safe.keyValue _ hy row (by omega) _ i.1.isLt _ i.2.1.isLt active.1 active.2).2⟩))
    · rw [if_neg active,if_neg active]
      rfl
  have scores : Source.scores M dot p s=KernelOnly.Specification.scores M dot p read
      (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) := by
    funext row i
    unfold Source.scores KernelOnly.Specification.scores
    by_cases active : i.2.1.val<p.length-row
    · rw [if_pos active,if_pos active,q,qs,k,ks row (by omega)]
      rfl
    · rw [if_neg active,if_neg active]
  have maximum : Source.maximum M dot p s=KernelOnly.Specification.maximum M dot p read
      (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) := by
    funext row acc i
    simp only [Source.maximum,KernelOnly.Specification.maximum,scores]
  have probability : Source.probability M dot p s=KernelOnly.Specification.probability M dot p read
      (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) := by
    funext row acc i
    simp only [Source.probability,KernelOnly.Specification.probability,scores,maximum]
  have rescale : Source.rescale M dot p s=KernelOnly.Specification.rescale M dot p read
      (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) := by
    funext row acc i
    simp only [Source.rescale,KernelOnly.Specification.rescale,maximum]
  have contribution : Source.contribution M dot p s=KernelOnly.Specification.contribution M dot p read
      (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) := by
    funext row acc i
    simp only [Source.contribution,KernelOnly.Specification.contribution,probability,v]
    rfl
  have step : Source.update M dot p s=KernelOnly.Specification.update M dot p read
      (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) := by
    funext row acc
    simp only [Source.update,KernelOnly.Specification.update,maximum,probability,rescale,contribution]
  unfold Whole.answer Source.outputValue Source.accumulated KernelOnly.Specification.algorithm
  rw [step]
  rfl

#axiomsClean algorithm
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.FiniteReference
