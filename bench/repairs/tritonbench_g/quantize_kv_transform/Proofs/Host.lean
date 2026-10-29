import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.Execution

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.Host
set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000

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

theorem checked_domain (p : Request) (heads : 0 < p.heads) (dimension : 0 < p.columns)
    (sources : p.tokens ≤ p.sourceRows)
    (logical_int32 : ∀ s, s < p.tokens → p.destination s < 2^31)
    (ok : reserve p.capacity [] (destinations p) = Bool.true) : Domain p :=
  ⟨heads, dimension, sources, (checked_destinations p ok).1, logical_int32, (checked_destinations p ok).2⟩

/-- Actual one-dimensional token grid and both head/column masks. -/
theorem exact_launch_coverage (p : Request) (s h c : Nat) :
    (s ∈ List.range p.tokens ∧ h ∈ List.range (headBlock p) ∧ h < p.heads ∧
      c ∈ List.range (columnBlock p) ∧ c < p.columns) ↔
    s < p.tokens ∧ h < p.heads ∧ c < p.columns := by
  simp only [List.mem_range]
  constructor
  · rintro ⟨hs,_,hh,_,hc⟩; exact ⟨hs,hh,hc⟩
  · rintro ⟨hs,hh,hc⟩
    exact ⟨hs,lt_of_lt_of_le hh (headBlock_covers p),hh,
      lt_of_lt_of_le hc (columnBlock_covers p),hc⟩

theorem source_input_address (p : Request) (v : Layout) (s h c : Nat) :
    inputAddress p v s h c = (v.tensor .input).data + 4*(s*(p.heads*p.columns)+h*p.columns+c) := by
  unfold inputAddress
  congr 1
  ring

theorem source_output_address (p : Request) (v : Layout) (slot h c : Nat) :
    outputAddress p v slot h c = (v.tensor .output).data + slot*(p.heads*p.columns)+h*p.columns+c := by
  unfold outputAddress
  ring

theorem source_scale_address (p : Request) (v : Layout) (slot h : Nat) (b : Fin 4) :
    scaleAddress p v slot h b = (v.tensor .scale).data + 4*(slot*p.heads+h)+b.val := by
  unfold scaleAddress
  omega

theorem four_byte_reconstruction (n : Nat) (bound : n < 4294967296) :
    n % 256 + (n / 256 % 256 * 256 +
    (n / 65536 % 256 * 65536 + (n / 16777216 % 256 * 16777216 + 0))) = n := by
  omega

/-- Four physical little-endian bytes reconstruct the actual checked index. -/
theorem metadata_from_int32 (p : Request) (d : Domain p) (v : Layout) (heap : Heap)
    (raw : RawMetadata p v heap) : Metadata p v heap := by
  intro s hs
  have hb : p.destination s < 4294967296 := lt_trans (d.destination_int32 s hs) (by norm_num)
  simp only [loadDest]
  rw [raw s hs 0 (by decide),raw s hs 1 (by decide),raw s hs 2 (by decide),raw s hs 3 (by decide)]
  simp only [int32Byte,pow_zero,pow_one,Nat.div_one,
    show (256 : Nat)^2 = 65536 by norm_num,
    show (256 : Nat)^3 = 16777216 by norm_num]
  have hr := four_byte_reconstruction (p.destination s) hb
  omega

/-- The accepted nonnegative INT32 word has the same value after sign extension. -/
theorem checked_int32_sign_extension (p : Request) (d : Domain p) (s : Nat) (hs : s < p.tokens) :
    (if p.destination s < 2^31 then (p.destination s : Int)
     else (p.destination s : Int) - 2^32) = (p.destination s : Int) := by
  rw [if_pos (d.destination_int32 s hs)]

def ascending (p : Request) : Schedule p := ⟨List.range p.tokens,List.Perm.refl _⟩

/-- Validation is completed before any Triton store. The rejected path keeps
the complete caller heap; None here represents the raised host error. -/
def guardedCall {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte) (v : Layout)
    (schedule : Schedule p) (heap : Heap) : Heap × Option (Option Unit) :=
  if reserve p.capacity [] (destinations p) then
    let result := call ops p plan decode encode v schedule heap
    (result.1, some result.2)
  else (heap, none)

theorem rejected_unchanged {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte) (v : Layout)
    (schedule : Schedule p) (heap : Heap) (rejected : reserve p.capacity [] (destinations p) = Bool.false) :
    guardedCall ops p plan decode encode v schedule heap = (heap, none) := by simp [guardedCall,rejected]

theorem accepted_call {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte) (v : Layout)
    (schedule : Schedule p) (heap : Heap) (ok : reserve p.capacity [] (destinations p) = Bool.true) :
    guardedCall ops p plan decode encode v schedule heap =
      ((call ops p plan decode encode v schedule heap).1, some none) := by simp [guardedCall,ok,call]

end VeriTile.Bench.Repairs.QuantizeKvTransform.Host

#guard_msgs (drop info) in
#auditModuleAxioms
