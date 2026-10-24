import bench.repairs.tritonbench_g.softmax_optimize.Proofs.DenseBytes

/-! Concrete sequential native word stores, including all four bytes, padding
and trailing allocation bytes. Repeated indices, if present, write the same
specified word; source schedules below separately establish unique coverage. -/
namespace VeriTile.Bench.Repairs.SoftmaxOptimize.NativeStores
open Bytes

def run (heap : Heap) (base : Nat) (values : Nat → Word) (indices : List Nat) : Heap :=
  storeMany heap (indices.map (fun i => (base + 4 * i, values i)))

theorem word_byte (heap : Heap) (base : Nat) (values : Nat → Word)
    (index written : Nat) (byte : Fin 4) :
    storeWord heap (base + 4 * written) (values written) (base + 4 * index + byte.val) =
      if index = written then encode (values index) byte else heap (base + 4 * index + byte.val) := by
  by_cases h : index = written
  · subst written
    rw [if_pos rfl]
    exact stored_byte heap _ _ byte
  · rw [if_neg h]
    apply store_frame
    have hb := byte.isLt
    omega

theorem byte (heap : Heap) (base : Nat) (values : Nat → Word) (indices : List Nat)
    (index : Nat) (offset : Fin 4) :
    run heap base values indices (base + 4 * index + offset.val) =
      if index ∈ indices then encode (values index) offset else heap (base + 4 * index + offset.val) := by
  induction indices generalizing heap with
  | nil => simp [run, storeMany]
  | cons head tail ih =>
    change run (storeWord heap (base + 4 * head) (values head)) base values tail _ = _
    rw [ih, word_byte]
    by_cases ht : index ∈ tail <;> by_cases hh : index = head <;> simp [ht, hh]

theorem readback (heap : Heap) (base : Nat) (values : Nat → Word) (indices : List Nat)
    (index : Nat) (member : index ∈ indices) :
    loadWord (run heap base values indices) (base + 4 * index) = values index := by
  unfold loadWord
  simp_rw [byte, if_pos member]
  exact decode_encode _

theorem frame (heap : Heap) (base : Nat) (values : Nat → Word) (indices : List Nat)
    (address : Nat) (outside : ∀ i ∈ indices, address < base + 4 * i ∨ base + 4 * i + 4 ≤ address) :
    run heap base values indices address = heap address := by
  apply storeMany_frame
  intro write hw
  rcases List.mem_map.mp hw with ⟨i, hi, rfl⟩
  exact outside i hi

theorem range_frame (heap : Heap) (base count : Nat) (values : Nat → Word) (address : Nat)
    (outside : address < base ∨ base + 4 * count ≤ address) :
    run heap base values (List.range count) address = heap address := by
  apply frame
  intro i hi
  have hi := List.mem_range.mp hi
  omega

theorem range_readback (heap : Heap) (base count : Nat) (values : Nat → Word)
    (index : Fin count) :
    loadWord (run heap base values (List.range count)) (base + 4 * index.val) = values index :=
  readback heap base values _ index (List.mem_range.mpr index.isLt)

/-- This identifies the implementation with the complete byte function already
used by ordinary copy APIs; unlike a postcondition-only model, run is literally
a fold of real little-endian storeWord updates. -/
theorem range_dense (heap : Heap) (base count : Nat) (values : Nat → Word) :
    run heap base values (List.range count) = DenseBytes.write heap base count (fun i => values i) := by
  funext address
  by_cases h : base ≤ address ∧ address < base + 4 * count
  · let index := (address - base) / 4
    let offset : Fin 4 := ⟨(address - base) % 4, Nat.mod_lt _ (by decide)⟩
    have hi : index < count := by dsimp [index]; omega
    have ha : address = base + 4 * index + offset.val := by dsimp [index, offset]; omega
    rw [ha, byte, if_pos (List.mem_range.mpr hi)]
    exact (DenseBytes.byte heap base count (fun i => values i) ⟨index, hi⟩ offset).symm
  · rw [range_frame _ _ _ _ _ (by omega), DenseBytes.frame _ _ _ _ _ (by omega)]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.NativeStores
