import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Numeric
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
namespace CompleteFrame

def Range (storage count : Nat) (p : Address) : Prop :=
  p.1=storage ∧ 0≤p.2 ∧ p.2<(count : Int)

def forwardWrites (s : Shape) (alloc : Allocation 3 caller) (final : Bool) (p : Address) : Prop :=
  Range (alloc.storage 0) (s.NK*rowCount s s.DV) p ∨
  (final=Bool.true ∧ Range (alloc.storage 1) (s.BH*s.DK*s.DV) p) ∨
  Range (alloc.storage 2) (rowCount s s.DV) p

def backwardWrites (s : Shape) (alloc : Allocation 6 caller) (p : Address) : Prop :=
  Range (alloc.storage 0) (s.NV*rowCount s s.DK) p ∨
  Range (alloc.storage 1) (s.NV*rowCount s s.DK) p ∨
  Range (alloc.storage 2) (s.NK*rowCount s s.DV) p ∨
  Range (alloc.storage 3) (rowCount s s.DK) p ∨
  Range (alloc.storage 4) (rowCount s s.DK) p ∨
  Range (alloc.storage 5) (rowCount s s.DV) p

theorem fill (storage count : Nat) (value : Nat → α) (m : Memory α) (p : Address)
    (off : ¬Range storage count p) : fillRange storage count value m p=m p := by
  have hn : ¬(p.1=storage ∧ 0≤p.2 ∧ p.2.toNat<count) := by
    intro h
    apply off
    refine ⟨h.1,h.2.1,?_⟩
    have cast : (p.2.toNat : Int)<count := Int.ofNat_lt.mpr h.2.2
    simpa only [Int.toNat_of_nonneg h.2.1] using cast
  simp only [fillRange,fill_get,List.mem_range,if_neg hn]

theorem forward (a : Arithmetic α) (s : Shape) (plans : Plans) (b : InputBuffers)
    (alloc : Allocation 3 caller) (seed final : Bool) (m : Memory α) (p : Address)
    (off : ¬forwardWrites s alloc final p) : sourceForward a s plans b alloc seed final m p=m p := by
  have h0 : ¬Range (alloc.storage 0) (s.NK*rowCount s s.DV) p := fun h=>off (Or.inl h)
  have h1 : final=Bool.true → ¬Range (alloc.storage 1) (s.BH*s.DK*s.DV) p :=
    fun hf h=>off (Or.inr (Or.inl ⟨hf,h⟩))
  have h2 : ¬Range (alloc.storage 2) (rowCount s s.DV) p := fun h=>off (Or.inr (Or.inr h))
  unfold sourceForward reduction sourcePhase
  rw [fill _ _ _ _ _ h2]
  cases final
  · exact fill _ _ _ _ _ h0
  · rw [if_pos rfl,fill _ _ _ _ _ (h1 rfl),fill _ _ _ _ _ h0]

theorem backward (a : Arithmetic α) (s : Shape) (plans : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α) (p : Address)
    (off : ¬backwardWrites s alloc p) : sourceBackward a s plans b alloc seed final m p=m p := by
  simp only [backwardWrites,not_or] at off
  unfold sourceBackward reduction sourcePhase
  rw [fill _ _ _ _ _ off.2.2.2.2.2,fill _ _ _ _ _ off.2.2.2.2.1,
    fill _ _ _ _ _ off.2.2.2.1,fill _ _ _ _ _ off.2.2.1,fill _ _ _ _ _ off.2.1,fill _ _ _ _ _ off.1]

/-- Byte offsets span every allocation and every unrelated storage, including
padding. The ordinary little-endian word codec leaves all other bytes intact. -/
abbrev Bytes := Nat × Int → BitVec 8

def byteView (m : Memory Numeric.Word) : Bytes := fun p=>
  BitVec.ofNat 8 ((m (p.1,p.2/4)).bits.toNat / 2^(8*(p.2%4).toNat))

theorem forward_bytes (a : Arithmetic Numeric.Word) (s : Shape) (plans : Plans) (b : InputBuffers)
    (alloc : Allocation 3 caller) (seed final : Bool) (m : Memory Numeric.Word) (p : Nat × Int)
    (off : ¬forwardWrites s alloc final (p.1,p.2/4)) :
    byteView (sourceForward a s plans b alloc seed final m) p=byteView m p := by
  simp only [byteView,forward a s plans b alloc seed final m _ off]

theorem backward_bytes (a : Arithmetic Numeric.Word) (s : Shape) (plans : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory Numeric.Word) (p : Nat × Int)
    (off : ¬backwardWrites s alloc (p.1,p.2/4)) :
    byteView (sourceBackward a s plans b alloc seed final m) p=byteView m p := by
  simp only [byteView,backward a s plans b alloc seed final m _ off]

#axiomsClean forward
#axiomsClean backward
#axiomsClean forward_bytes
#axiomsClean backward_bytes
end CompleteFrame
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
