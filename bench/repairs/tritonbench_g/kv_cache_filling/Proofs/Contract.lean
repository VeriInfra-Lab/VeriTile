import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Host
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KvCacheFilling

theorem checkDomain_iff (r : Request) : checkDomain r = true ↔ Domain r := by
  simp only [checkDomain,Bool.and_eq_true,decide_eq_true_eq]
  constructor
  · rintro ⟨⟨⟨⟨⟨⟨⟨⟨hh,hb⟩,hk⟩,hv⟩,ks⟩,vs⟩,lengths⟩,pages⟩,owned⟩
    refine ⟨hh,hb,hk,hv,ks,vs,?_,?_,?_⟩
    · intro b hb
      exact lengths ⟨b,hb⟩
    · intro b t hb ht
      exact pages ⟨b,hb⟩ ⟨t,ht⟩
    · intro b t b' t' hb ht hb' ht' eq
      exact owned ⟨b,hb⟩ ⟨t,ht⟩ ⟨b',hb'⟩ ⟨t',ht'⟩ eq
  · intro h
    refine ⟨⟨⟨⟨⟨⟨⟨⟨h.heads_pos,h.block_pos⟩,h.dk_pos⟩,h.dv_pos⟩,h.key_shape⟩,h.value_shape⟩,?_⟩,?_⟩,?_⟩
    · intro b; exact h.lengths b b.isLt
    · intro b t; exact h.page_bounds b t b.isLt t.isLt
    · intro b t b' t' eq
      exact h.token_ownership b t b' t' b.isLt t.isLt b'.isLt t'.isLt eq

theorem reserve_iff (xs used : List Nat) (hn : used.Nodup) :
    (∃ final,reserve xs used=some final) ↔ xs.Nodup ∧ xs.Disjoint used := by
  induction xs generalizing used with
  | nil => simp [reserve]
  | cons x xs ih =>
    simp only [reserve]
    split_ifs with hx
    · simp only [reduceCtorEq,exists_false, false_iff,not_and]
      intro _ hd
      exact List.disjoint_left.mp hd (List.mem_cons_self) hx
    · rw [ih (x::used) (List.nodup_cons.mpr ⟨hx,hn⟩)]
      constructor
      · rintro ⟨ht,hd⟩
        have hxt : x∉xs := fun mem=>List.disjoint_left.mp hd mem (List.mem_cons_self)
        refine ⟨List.nodup_cons.mpr ⟨hxt,ht⟩,?_⟩
        apply List.disjoint_left.mpr
        intro a ha hau
        rcases List.mem_cons.mp ha with eq|mem
        · subst a; contradiction
        · exact List.disjoint_left.mp hd mem (List.mem_cons_of_mem _ hau)
      · rintro ⟨ht,hd⟩
        obtain ⟨hxt,ht⟩ := List.nodup_cons.mp ht
        refine ⟨ht,?_⟩
        apply List.disjoint_left.mpr
        intro a ha hau
        rcases List.mem_cons.mp hau with eq|mem
        · subst a; contradiction
        · exact List.disjoint_left.mp hd (List.mem_cons_of_mem _ ha) mem

theorem checkInputs_iff (t : TensorInputs) (r : Request) :
    checkInputs t r=true ↔ InputContract t r := by
  simp only [checkInputs,Bool.and_eq_true,decide_eq_true_eq,checkDomain_iff,InputContract,and_assoc]

/-- Signed metadata is converted to Nat only after the exact nonnegative
check. This rules out totalizing a negative int32 value into row zero. -/
theorem checked_signed_value (i : Int) (h : 0 ≤ i) : (i.toNat : Int)=i := Int.toNat_of_nonneg h

theorem checkAdmission_iff (t : TensorInputs) (dimensions : Request) (s : SignedMetadata) :
    checkAdmission t dimensions s=true ↔ Admission t dimensions s := by
  simp only [checkAdmission,Bool.and_eq_true,checkInputs_iff,decide_eq_true_eq,Admission]

/-- The natural schedule contains exactly the admitted signed metadata, not
the zero produced by a totalized conversion of a negative value. -/
theorem admitted_metadata (t : TensorInputs) (dimensions : Request) (s : SignedMetadata)
    (h : Admission t dimensions s) (b : Fin dimensions.batches) :
    ((s.request dimensions).start b : Int)=s.start b ∧
    ((s.request dimensions).count b : Int)=s.count b ∧
    ((s.request dimensions).total b : Int)=s.total b := by
  obtain ⟨hs,hc,ht⟩ := h.2.1 b
  exact ⟨checked_signed_value _ hs,checked_signed_value _ hc,checked_signed_value _ ht⟩

theorem admitted_page (t : TensorInputs) (dimensions : Request) (s : SignedMetadata)
    (h : Admission t dimensions s) (b : Fin dimensions.batches)
    (i : Fin ((s.request dimensions).count b)) :
    let r := s.request dimensions
    (r.page b ((r.history b+i.val)/r.block) : Int)=s.page b ((r.history b+i.val)/r.block) :=
  checked_signed_value _ (h.2.2 b i)

#axiomsClean checkInputs_iff
#axiomsClean checkDomain_iff
#axiomsClean reserve_iff
#axiomsClean checkAdmission_iff
#axiomsClean admitted_metadata
#axiomsClean admitted_page
end VeriTile.Bench.Repairs.KvCacheFilling
