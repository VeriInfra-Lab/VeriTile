import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.Execution

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.Host
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

def destinations (p : Request) : List Nat := (List.range p.tokens).map p.slot

theorem checked_destinations (p : Request) (ok : reserve p.capacity [] (destinations p) = Bool.true) :
    (∀ s, s < p.tokens → p.slot s < p.capacity) ∧
    (∀ s t, s < p.tokens → t < p.tokens → p.slot s = p.slot t → s = t) := by
  obtain ⟨bounds, unique⟩ := (reserve_iff p.capacity (destinations p) []).mp ok
  constructor
  · intro s hs
    exact (bounds _ (List.mem_map.mpr ⟨s,List.mem_range.mpr hs,rfl⟩)).1
  · intro s t hs ht eq
    exact List.inj_on_of_nodup_map unique (List.mem_range.mpr hs) (List.mem_range.mpr ht) eq

theorem checked_domain (p : Request) (heads : 0 < p.headCount) (groups : 0 < p.groups)
    (sources : p.tokens ≤ p.sourceRows)
    (logical_int32 : ∀ s, s < p.tokens → p.slot s < 2^31)
    (ok : reserve p.capacity [] (destinations p) = Bool.true) : Domain p :=
  ⟨heads, groups, sources, (checked_destinations p ok).1, logical_int32, (checked_destinations p ok).2⟩

/-- A bijective encoding of the actual two-dimensional program ID. -/
def flatten (p : Request) (token head : Nat) : Nat := token * p.headCount + head

theorem flatten_bound (p : Request) (token head : Nat) (ht : token < p.tokens) (hh : head < p.headCount) :
    flatten p token head < p.programs := RW.linear_bound p.tokens p.headCount token head ht hh

theorem flatten_injective (p : Request) (s t h k : Nat) (hh : h < p.headCount) (hk : k < p.headCount)
    (same : flatten p s h = flatten p t k) : s = t ∧ h = k :=
  RW.linear_injective p.headCount s t h k hh hk same

theorem unflatten (p : Request) (d : Domain p) (token head : Nat) (hh : head < p.headCount) :
    flatten p token head / p.headCount = token ∧ flatten p token head % p.headCount = head := by
  constructor
  · simpa [flatten,Nat.add_comm,Nat.div_eq_of_lt hh] using
      Nat.add_mul_div_right head token d.heads_positive
  · simp [flatten,Nat.add_mod,Nat.mod_eq_of_lt hh]

theorem flatten_complete (p : Request) (d : Domain p) (program : Nat) (hp : program < p.programs) :
    ∃ token head, token < p.tokens ∧ head < p.headCount ∧ flatten p token head = program := by
  exact ⟨program / p.headCount, program % p.headCount, program_token_bound p program hp,
    Nat.mod_lt _ d.heads_positive, by simpa [flatten,Nat.mul_comm] using Nat.div_add_mod program p.headCount⟩

/-- Actual grid=(tokens,heads), group padding/mask and exactly eight lanes. -/
theorem exact_launch_coverage (p : Request) (token head group lane : Nat) :
    (token ∈ List.range p.tokens ∧ head ∈ List.range p.headCount ∧
      group ∈ List.range (groupBlock p) ∧ group < p.groups ∧ lane ∈ List.range groupWidth) ↔
    token < p.tokens ∧ head < p.headCount ∧ group < p.groups ∧ lane < groupWidth := by
  simp only [List.mem_range]
  constructor
  · rintro ⟨ht,hh,_,hg,hc⟩; exact ⟨ht,hh,hg,hc⟩
  · rintro ⟨ht,hh,hg,hc⟩; exact ⟨ht,hh,lt_of_lt_of_le hg (groupBlock_covers p),hg,hc⟩

/-- The split view shares storage; these equalities identify the exact native
row-major addresses for both launch coordinates, all groups and lanes. -/
theorem source_input_address (p : Request) (v : Layout) (token head group lane : Nat) :
    inputAddress p v (flatten p token head) group lane =
      (v.tensor .input).data + 4 * (token*(p.headCount*p.groups*8) + head*(p.groups*8) + group*8 + lane) := by
  simp only [inputAddress,flatten,groupWidth]
  congr 1
  ring

theorem source_output_address (p : Request) (d : Domain p) (v : Layout)
    (token head group lane : Nat) (hh : head < p.headCount) :
    outputAddress p v (p.destination (flatten p token head)) group lane =
      (v.tensor .output).data + (p.slot token*(p.headCount*p.groups*8) + head*(p.groups*8) + group*8 + lane) := by
  simp only [Request.destination,(unflatten p d token head hh).1,(unflatten p d token head hh).2,outputAddress,groupWidth]
  congr 1
  ring

theorem source_scale_address (p : Request) (d : Domain p) (v : Layout)
    (token head group : Nat) (b : Fin 4) (hh : head < p.headCount) :
    scaleAddress p v (p.destination (flatten p token head)) group b =
      (v.tensor .scale).data + 4*(p.slot token*(p.headCount*p.groups) + head*p.groups + group) + b.val := by
  simp only [Request.destination,(unflatten p d token head hh).1,(unflatten p d token head hh).2,scaleAddress]
  ring

theorem metadata_from_int32 (p : Request) (v : Layout) (heap : Heap) (raw : RawMetadata p v heap) :
    Metadata p v heap := by
  intro s hs
  simp only [loadTarget,destAddress,raw _ (program_token_bound p s hs),Request.destination]

theorem destination_flatten (p : Request) (d : Domain p) (token head : Nat) (hh : head < p.headCount) :
    p.destination (flatten p token head) = p.slot token*p.headCount+head := by
  simp only [Request.destination,(unflatten p d token head hh).1,(unflatten p d token head hh).2]

theorem touched_iff_logical (p : Request) (d : Domain p) (v : Layout) (a : Nat) :
    Touched p v a ↔ LogicalTouched p v a := by
  constructor
  · rintro (⟨s,g,c,hs,hg,hc,ha⟩ | ⟨s,g,b,hs,hg,ha⟩)
    · exact Or.inl ⟨s/p.headCount,s%p.headCount,g,c,program_token_bound p s hs,
        Nat.mod_lt _ d.heads_positive,hg,hc,ha⟩
    · exact Or.inr ⟨s/p.headCount,s%p.headCount,g,b,program_token_bound p s hs,
        Nat.mod_lt _ d.heads_positive,hg,ha⟩
  · rintro (⟨t,h,g,c,ht,hh,hg,hc,ha⟩ | ⟨t,h,g,b,ht,hh,hg,ha⟩)
    · refine Or.inl ⟨flatten p t h,g,c,flatten_bound p t h ht hh,hg,hc,?_⟩
      simpa only [destination_flatten p d t h hh,logicalOutputAddress] using ha
    · refine Or.inr ⟨flatten p t h,g,b,flatten_bound p t h ht hh,hg,?_⟩
      simpa only [destination_flatten p d t h hh,logicalScaleAddress] using ha

def ascending (p : Request) : Schedule p := ⟨List.range p.programs,List.Perm.refl _⟩

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

end VeriTile.Bench.Repairs.QuantizeKvCopy.Host

#guard_msgs (drop info) in
#auditModuleAxioms
