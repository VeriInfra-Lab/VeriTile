import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.NativeControl
open NativeIndex Storage

/-- Local emitted-program facts: CUDA PIDs and Triton's arange lanes are
signed32 before the explicit source widening. These are compiler ABI facts;
they do not add a host tensor-shape admission check. -/
structure ABI (h : Call) where
  program : NativeIndex.Program h
  arangeBound : h.BK≤2147483648

theorem arange_widens (h : Call) (a : ABI h) (k : Fin h.BK) (n : Fin 256) :
    BitVec.ofInt 64 (BitVec.ofNat 32 k.val).toInt=BitVec.ofNat 64 k.val ∧
    BitVec.ofInt 64 (BitVec.ofNat 32 n.val).toInt=BitVec.ofNat 64 n.val := by
  exact ⟨widen32 _ (k.isLt.trans_le a.arangeBound),widen32 _ (by have := n.isLt; omega)⟩

theorem metadata_widens (word : BitVec 32) :
    (BitVec.ofInt 64 word.toInt).toInt=word.toInt := by
  have lo := word.le_toInt
  have hi := word.toInt_lt
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · norm_num at lo ⊢; omega
  · norm_num at hi ⊢; omega

/-- The actual same-call padded allocation entails safe signed64 cdiv
addition, subtraction, division and a one-iteration source loop. -/
theorem cdiv_bound (h : Call) (s : Placement h) (p : Geometry.Program h) :
    h.N+h.N<signedLimit ∧ (h.N+h.N-1)/h.N=1 := by
  have batchPositive : 0<h.batches := by have := p.batchInside; omega
  have capacity : h.columns+255≤s.output.extent := by
    rw [s.outputSize]
    exact Nat.le_mul_of_pos_left _ batchPositive
  have scaled := Nat.mul_le_mul_right 4 capacity
  have size := s.output.signedSize
  rw [s.outputWidth] at size
  have slice := h.sliceInside
  refine ⟨?_,Kernels.split_length h (by have := p.rowInside; omega)⟩
  omega

def cdivNumerator (h : Call) : BitVec 64 :=
  BitVec.ofNat 64 h.N+BitVec.ofNat 64 h.N-1

theorem cdiv_word (h : Call) (s : Placement h) (p : Geometry.Program h) :
    (cdivNumerator h).toInt=(h.N+h.N-1:Int) ∧
    (BitVec.ofNat 64 h.N).toInt=(h.N:Int) := by
  obtain ⟨bound,_⟩ := cdiv_bound h s p
  have pos : 0<h.N := by have := p.rowInside; omega
  have enough : 1≤h.N+h.N := by omega
  constructor
  · unfold cdivNumerator
    rw [←BitVec.ofNat_add]
    change (BitVec.ofNat 64 (h.N+h.N)-BitVec.ofNat 64 1).toInt=_
    rw [BitVec.ofNat_sub_ofNat_of_le _ _ (by decide) enough]
    simpa [Int.natCast_sub enough] using signed64 (h.N+h.N-1) (by omega)
  · exact signed64 _ (by omega)

theorem metadata_address (h : Call) (s : Placement h) (p : NativeIndex.Program h) :
    (BitVec.ofNat 64 s.indices.base+
      BitVec.ofInt 64 (BitVec.ofNat 32 p.batch).toInt*BitVec.ofNat 64 s.indices.bytes).toNat=
      s.indices.address p.batch := by
  rw [widen32 p.batch p.batchSigned32]
  exact pointer_word s.indices p.batch (by rw [s.indexSize]; exact p.batchInside)

theorem loop_mask (n : Fin 256) : (n.val<1 ↔ n.val=0) := by omega

#axiomsClean arange_widens
#axiomsClean metadata_widens
#axiomsClean cdiv_word
#axiomsClean metadata_address
end VeriTile.Bench.Repairs.BgmvExpandSlice.NativeControl
