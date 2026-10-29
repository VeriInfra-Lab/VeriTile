import bench.repairs.tritonbench_g.destindex_copy_kv1.Contract
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.Grid

namespace VeriTile.Bench.Repairs.DestindexCopyKv1
noncomputable section

theorem checkInputs_iff (m : Inputs) (words : Nat → Word) :
    checkInputs m words=.ok () ↔ InputContract m words := by
  have fields : InputContract m words ↔
    (∀ r, (m r).layout=.strided) ∧ Ranks m ∧ Types m ∧
    (∀ r, (m r).device.kind="cuda") ∧ (∀ r, (m r).device=(m .source).device) ∧
    (∀ r, (m r).isContiguous=Bool.true) ∧ (∀ r, (m r).isNeg=Bool.false) ∧ Shapes m ∧
    (∀ r, (m r).numel<2^31) ∧ block (config m).heads * block (config m).dims≤2^20 ∧
    (∀ r, 0<(m r).numel → base m r%4=0) ∧ Separate m ∧ Bounds m words ∧ Unique m words := by
    constructor
    · intro h; exact ⟨h.layout,h.rank,h.dtype,h.cuda,h.device,h.contiguous,h.noNeg,h.shapes,
        h.extent,h.tile,h.alignment,h.separate,h.bounds,h.unique⟩
    · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l,n,o⟩; exact ⟨a,b,c,d,e,f,g,h,i,j,k,l,n,o⟩
  rw [fields]
  simp [checkInputs,ite_eq_iff]

theorem checkArguments_iff (m : Option Inputs) (words : Nat → Word) :
    checkArguments m words=.ok () ↔ Admitted m words := by
  cases m <;> simp [checkArguments,Admitted,checkInputs_iff]

theorem triple (xs : List Nat) (h : xs.length=3) :
    xs=[xs.getD 0 0,xs.getD 1 0,xs.getD 2 0] := by
  rcases xs with _ | ⟨a,xs⟩ <;> simp at h ⊢
  rcases xs with _ | ⟨b,xs⟩ <;> simp at h ⊢
  rcases xs with _ | ⟨c,xs⟩ <;> simp at h ⊢
  simpa using h

theorem dimensions (m : Inputs) (w : Nat → Word) (h : InputContract m w) :
    (m .source).numel=(config m).sourceRows*(config m).heads*(config m).dims ∧
    (m .output).numel=(config m).rows*(config m).heads*(config m).dims := by
  have hs := triple (m .source).shape h.rank.1
  have ho := triple (m .output).shape h.rank.2.1
  have tail := h.shapes.1
  rw [hs,ho] at tail
  simp only [List.drop_succ_cons,List.drop_zero,List.cons.injEq,and_true] at tail
  constructor
  · change (m .source).shape.prod=_
    rw [hs]
    simp [config,Nat.mul_assoc]
  · change (m .output).shape.prod=_
    rw [ho]
    simp only [config,List.prod_cons,List.prod_nil,Nat.mul_one]
    rw [tail.1,tail.2]
    ring

theorem destination_bound (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (p : Nat) (hp : p<(config m).count) : destination w p<(config m).rows := by
  have hw := h.bounds p hp
  unfold destination
  omega

theorem output_bound (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (p a b : Nat) (hp : p<(config m).count) (ha : a<(config m).heads) (hb : b<(config m).dims) :
    flat (config m) (destination w p) a b<(m .output).numel := by
  rw [(dimensions m w h).2]
  exact Grid.flat_bound _ _ _ _ _ (destination_bound m w h p hp) ha hb

theorem source_bound (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (p a b : Nat) (hp : p<(config m).count) (ha : a<(config m).heads) (hb : b<(config m).dims) :
    flat (config m) p a b<(m .source).numel := by
  rw [(dimensions m w h).1]
  exact Grid.flat_bound _ _ _ _ _ (lt_of_lt_of_le hp h.shapes.2) ha hb

#axiomsClean checkArguments_iff
#axiomsClean dimensions
#axiomsClean output_bound
#axiomsClean source_bound
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1
