import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.Execution

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.Host
set_option autoImplicit false
set_option maxHeartbeats 2000000

/-- The actual occupied-set loop, represented as its membership-equivalent list. -/
def reserve (capacity : Nat) : List Nat → List Nat → Bool
  | _, [] => Bool.true
  | seen, x :: xs => if x < capacity ∧ x ∉ seen then reserve capacity (x :: seen) xs else Bool.false

theorem reserve_iff (capacity : Nat) (xs seen : List Nat) :
    reserve capacity seen xs = Bool.true ↔
      (∀ x ∈ xs, x < capacity ∧ x ∉ seen) ∧ xs.Nodup := by
  induction xs generalizing seen with
  | nil => simp [reserve]
  | cons x xs ih =>
    by_cases hx : x < capacity ∧ x ∉ seen
    · simp only [reserve, if_pos hx, ih, List.nodup_cons, List.mem_cons]
      constructor
      · rintro ⟨rest, nodup⟩
        refine ⟨?_, ?_, nodup⟩
        · intro y hy
          rcases hy with rfl | hy
          · exact hx
          · have hr := rest y hy
            exact ⟨hr.1, fun hyseen => hr.2 (Or.inr hyseen)⟩
        · intro hxx
          exact (rest x hxx).2 (Or.inl rfl)
      · rintro ⟨all, absent, nodup⟩
        refine ⟨?_,nodup⟩
        intro y hy
        have hy' := all y (Or.inr hy)
        refine ⟨hy'.1,?_⟩
        rintro (rfl | hyseen)
        · exact absent hy
        · exact hy'.2 hyseen
    · simp only [reserve, if_neg hx, Bool.false_eq_true, false_iff]
      rintro ⟨all,_⟩
      exact hx (all x (by simp))

def destinations (p : Request) : List Nat := (List.range p.tokens).map p.destination

theorem checked_destinations (p : Request) (ok : reserve p.capacity [] (destinations p) = Bool.true) :
    (∀ s, s < p.tokens → p.destination s < p.capacity) ∧
    (∀ s t, s < p.tokens → t < p.tokens → p.destination s = p.destination t → s = t) := by
  obtain ⟨bounds, unique⟩ := (reserve_iff p.capacity (destinations p) []).mp ok
  constructor
  · intro s hs
    exact (bounds _ (List.mem_map.mpr ⟨s,List.mem_range.mpr hs,rfl⟩)).1
  · intro s t hs ht eq
    exact List.inj_on_of_nodup_map unique (List.mem_range.mpr hs) (List.mem_range.mpr ht) eq

theorem checked_domain (p : Request) (heads : 0 < p.heads) (dimension : ∃ n, p.columns = 2^n)
    (sources : p.tokens ≤ p.sourceRows)
    (logical_int64 : ∀ s, s < p.tokens → p.destination s < 2^63)
    (ok : reserve p.capacity [] (destinations p) = Bool.true) : Domain p :=
  ⟨heads, dimension, sources, (checked_destinations p ok).1, logical_int64, (checked_destinations p ok).2⟩

/-- grid=(seq_len,), and the actual source mask covers precisely all real heads.
There is no unproved launch-to-logical-coordinate assumption here. -/
theorem exact_launch_coverage (p : Request) (s h c : Nat) :
    (s ∈ List.range p.tokens ∧ h ∈ List.range (headBlock p) ∧ h < p.heads ∧ c ∈ List.range p.columns) ↔
    s < p.tokens ∧ h < p.heads ∧ c < p.columns := by
  simp only [List.mem_range]
  constructor
  · rintro ⟨hs,_,hh,hc⟩; exact ⟨hs,hh,hc⟩
  · rintro ⟨hs,hh,hc⟩; exact ⟨hs,lt_of_lt_of_le hh (headBlock_covers p),hh,hc⟩

def ascending (p : Request) : Schedule p := ⟨List.range p.tokens,List.Perm.refl _⟩

/-- Validation is completed before any Triton store. The rejected path keeps
the complete caller heap; None here represents the raised host error. -/
def guardedCall {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout)
    (schedule : Schedule p) (heap : Heap) : Heap × Option (Option Unit) :=
  if reserve p.capacity [] (destinations p) then
    let result := call ops p plan decode encode v schedule heap
    (result.1, some result.2)
  else (heap, none)

theorem rejected_unchanged {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout)
    (schedule : Schedule p) (heap : Heap) (rejected : reserve p.capacity [] (destinations p) = Bool.false) :
    guardedCall ops p plan decode encode v schedule heap = (heap, none) := by simp [guardedCall,rejected]

theorem accepted_call {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout)
    (schedule : Schedule p) (heap : Heap) (ok : reserve p.capacity [] (destinations p) = Bool.true) :
    guardedCall ops p plan decode encode v schedule heap =
      ((call ops p plan decode encode v schedule heap).1, some none) := by simp [guardedCall,ok,call]

end VeriTile.Bench.Repairs.QuantizeCopyKv.Host

#guard_msgs (drop info) in
#auditModuleAxioms
