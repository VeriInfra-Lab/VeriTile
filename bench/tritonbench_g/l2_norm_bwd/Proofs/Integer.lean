import bench.tritonbench_g.l2_norm_bwd.Proofs.Physical
import bench.tritonbench_g.swiglu_triton.Proofs.Index

namespace VeriTile.Bench.Repairs.L2NormBwd.Integer
open Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Index (signed_ofNat)
noncomputable section
set_option maxHeartbeats 4000000

/-- Ordinary signed-64 PyTorch stride metadata, not an output/address equation. -/
structure Metadata (m : Matrix) : Prop where
  row : m.rowStride<2^63
  col : m.colStride<2^63

/-- The only grid boundary retained for the ordinary launcher: its actual
one-dimensional program IDs must be representable in the native PID field. -/
def GridCompatible (rows : Nat) : Prop := rows≤2^31

def rowProduct (row stride : Nat) : BitVec 64 := BitVec.ofNat 64 row*BitVec.ofNat 64 stride

def pointer (m : Matrix) (row col : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (m.view.storage.baseByte+4*m.view.storageOffset)+
    rowProduct row m.rowStride*BitVec.ofNat 64 4+
    (BitVec.ofNat 64 col*BitVec.ofNat 64 m.colStride)*BitVec.ofNat 64 4

theorem mask_exact (N B col : Nat) (hN : N≤B) (hB : B≤16384) (hc : col<B) :
    (BitVec.ofNat 64 col).slt (BitVec.ofNat 64 N)=decide (col<N) := by
  change decide ((BitVec.ofNat 64 col).toInt<(BitVec.ofNat 64 N).toInt)=_
  rw [signed_ofNat 64 col (by decide) (by omega),signed_ofNat 64 N (by decide) (by omega)]
  simp only [Int.ofNat_lt]

/-- Bounds follow from the actual backed active coordinate. Even a huge stride
on a singleton dimension is allowed; inactive padded pointers are modular and
are never dereferenced, so no false padding-allocation premise is introduced. -/
theorem active_no_wrap_of_pid (m : Matrix) (metadata : Metadata m) (backed : m.backed)
    (r c : Nat) (rbound : r<2^32) (hr : r<m.rows) (hc : c<m.cols) :
    (BitVec.ofNat 32 r).toNat=r ∧
    (BitVec.ofNat 64 m.rowStride).toInt=(m.rowStride:Int) ∧
    (BitVec.ofNat 64 m.colStride).toInt=(m.colStride:Int) ∧
    (rowProduct r m.rowStride).toInt=(r*m.rowStride:Nat) ∧
    (BitVec.ofNat 64 c*BitVec.ofNat 64 m.colStride).toInt=(c*m.colStride:Nat) ∧
    (pointer m r c).toNat=(m.key r c).address ∧
    ∀ byte,byte<4 → (pointer m r c).toNat+byte<m.view.storage.baseByte+m.view.storage.bytes ∧
      (pointer m r c).toNat+byte<2^64 := by
  obtain ⟨_,lower,upper⟩ := backed r c hr hc
  have capacity := m.view.storage.representable
  simp only [Matrix.key,Matrix.index,Allocation.key] at lower upper
  have rmul : r*m.rowStride<2^63 := by omega
  have cmul : c*m.colStride<2^63 := by omega
  have arith : m.view.storage.baseByte+4*m.view.storageOffset+r*m.rowStride*4+c*m.colStride*4=
      m.view.storage.baseByte+4*(m.view.storageOffset+r*m.rowStride+c*m.colStride) := by ring
  have exactPointer : (pointer m r c).toNat=(m.key r c).address := by
    unfold pointer rowProduct
    rw [←BitVec.ofNat_mul,←BitVec.ofNat_mul,←BitVec.ofNat_mul,←BitVec.ofNat_mul,
      ←BitVec.ofNat_add,←BitVec.ofNat_add,BitVec.toNat_ofNat,arith,Nat.mod_eq_of_lt (by omega)]
    rfl
  refine ⟨by rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt rbound],
    signed_ofNat 64 _ (by decide) metadata.row,signed_ofNat 64 _ (by decide) metadata.col,
    ?_,?_,exactPointer,?_⟩
  · rw [rowProduct,←BitVec.ofNat_mul];exact signed_ofNat 64 _ (by decide) rmul
  · rw [←BitVec.ofNat_mul];exact signed_ofNat 64 _ (by decide) cmul
  · intro byte small
    rw [exactPointer]
    simp only [Matrix.key,Matrix.index,Allocation.key]
    omega

theorem active_no_wrap (m : Matrix) (metadata : Metadata m) (backed : m.backed)
    (grid : GridCompatible m.rows) (r c : Nat) (hr : r<m.rows) (hc : c<m.cols) :
    (BitVec.ofNat 32 r).toNat=r ∧
    (BitVec.ofNat 64 m.rowStride).toInt=(m.rowStride:Int) ∧
    (BitVec.ofNat 64 m.colStride).toInt=(m.colStride:Int) ∧
    (rowProduct r m.rowStride).toInt=(r*m.rowStride:Nat) ∧
    (BitVec.ofNat 64 c*BitVec.ofNat 64 m.colStride).toInt=(c*m.colStride:Nat) ∧
    (pointer m r c).toNat=(m.key r c).address ∧
    ∀ byte,byte<4 → (pointer m r c).toNat+byte<m.view.storage.baseByte+m.view.storage.bytes ∧
      (pointer m r c).toNat+byte<2^64 := by
  exact active_no_wrap_of_pid m metadata backed r c (by unfold GridCompatible at grid;omega) hr hc

#axiomsClean active_no_wrap
#axiomsClean mask_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Integer
