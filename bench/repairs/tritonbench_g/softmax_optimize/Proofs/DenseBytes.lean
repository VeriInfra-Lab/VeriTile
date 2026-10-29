import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Bytes

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.DenseBytes
open Bytes

/-- A concrete coordinate-copy implementation for ordinary host tensor APIs.
It writes each complete encoded FP32 word, including all NaN and signed-zero bits. -/
def write (heap : Heap) (base count : Nat) (values : Fin count → Word) : Heap := fun address =>
  if h : base ≤ address ∧ address < base + 4 * count then
    encode (values ⟨(address - base) / 4, by omega⟩)
      ⟨(address - base) % 4, Nat.mod_lt _ (by decide)⟩
  else heap address

theorem byte (heap : Heap) (base count : Nat) (values : Fin count → Word)
    (index : Fin count) (offset : Fin 4) :
    write heap base count values (base + 4 * index.val + offset.val) =
      encode (values index) offset := by
  unfold write
  rw [dif_pos (by have hi := index.isLt; have hb := offset.isLt; omega)]
  have hindex : (base + 4 * index.val + offset.val - base) / 4 = index.val := by
    have hb := offset.isLt
    omega
  have hoffset : (base + 4 * index.val + offset.val - base) % 4 = offset.val := by
    have hb := offset.isLt
    omega
  congr 2 <;> apply Fin.ext <;> assumption

theorem readback (heap : Heap) (base count : Nat) (values : Fin count → Word) (index : Fin count) :
    loadWord (write heap base count values) (base + 4 * index.val) = values index := by
  unfold loadWord
  simp_rw [byte]
  exact decode_encode _

theorem frame (heap : Heap) (base count : Nat) (values : Fin count → Word) (address : Nat)
    (outside : address < base ∨ base + 4 * count ≤ address) :
    write heap base count values address = heap address := by
  unfold write
  rw [dif_neg (by omega)]

theorem empty (heap : Heap) (base : Nat) (values : Fin 0 → Word) : write heap base 0 values = heap := by
  funext address
  apply frame
  omega

theorem fresh_frame (heap : Heap) (live : List Span) (output : Span)
    (fresh : ∀ span ∈ live, span.disjoint output) (count : Nat) (values : Fin count → Word)
    (holds : 4 * count ≤ output.bytes) :
    Frame live heap (write heap output.base count values) := by
  intro span hspan address ha
  apply frame
  have hd := fresh span hspan
  unfold Span.contains at ha
  unfold Span.disjoint at hd
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.DenseBytes
