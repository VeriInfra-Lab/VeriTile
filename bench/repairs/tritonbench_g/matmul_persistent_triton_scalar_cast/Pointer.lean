import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Values
import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.ControlBounds

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Pointer
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open Core
set_option maxHeartbeats 5000000

/-- All nine public scalar casts preserve ordinary nonnegative i64 metadata.
No special case excludes scalar one or zero-stride broadcasting. -/
theorem scalar_bounds (p : Parameters) (b : Buffers) (layout : Layout p.problem b)
    (strides : Strides p b) :
    p.problem.M<2^63 ∧ p.problem.N<2^63 ∧ p.problem.K<2^63 ∧
    p.am<2^63 ∧ p.ak<2^63 ∧ p.bk<2^63 ∧ p.bn<2^63 ∧ p.cm<2^63 ∧ p.cn<2^63 := by
  exact ⟨layout.aShape.1 ▸ layout.aBound.rows,layout.bShape.2 ▸ layout.bBound.columns,
    layout.aShape.2 ▸ layout.aBound.columns,strides.am ▸ layout.aBound.rowStride,
    strides.ak ▸ layout.aBound.columnStride,strides.bk ▸ layout.bBound.rowStride,
    strides.bn ▸ layout.bBound.columnStride,strides.cm ▸ layout.cBound.rowStride,
    strides.cn ▸ layout.cBound.columnStride⟩

def bytePointer (tensor : Tensor) (offset : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (tensor.pointer.toNat+tensor.dtype.bytes*offset)

theorem bytePointer_exact (tensor : Tensor) (bounded : tensor.Bounded)
    (row column : Nat) (hr : row<tensor.rows) (hc : column<tensor.columns) :
    (bytePointer tensor (row*tensor.rowStride+column*tensor.columnStride)).toNat=
      tensor.pointer.toNat+tensor.dtype.bytes*(row*tensor.rowStride+column*tensor.columnStride) := by
  simp only [bytePointer,BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (tensor.pointer_products bounded row column hr hc).2.2.2.2

/-- The offset used by the executed DSL is exactly the offset at the actual
Tensor pointer. Clamped inactive lanes are included. -/
theorem a_offset_pointer (p : Parameters) (b : Buffers) (layout : Layout p.problem b)
    (strides : Strides p b) (hm : 0<p.problem.M) (hk : 0<p.problem.K)
    (tile block : Nat) (i : AIndex p) :
    (bytePointer b.A (aOffset p tile block i)).toNat=
      b.A.pointer.toNat+b.A.dtype.bytes*(aOffset p tile block i) := by
  rw [aOffset,strides.am,strides.ak]
  exact bytePointer_exact b.A layout.aBound _ _
    (by rw [layout.aShape.1]; exact clamp_bound _ _ hm)
    (by rw [layout.aShape.2]; exact clamp_bound _ _ hk)

theorem b_offset_pointer (p : Parameters) (b : Buffers) (layout : Layout p.problem b)
    (strides : Strides p b) (hn : 0<p.problem.N) (hk : 0<p.problem.K)
    (tile block : Nat) (i : BIndex p) :
    (bytePointer b.B (bOffset p tile block i)).toNat=
      b.B.pointer.toNat+b.B.dtype.bytes*(bOffset p tile block i) := by
  rw [bOffset,strides.bk,strides.bn]
  exact bytePointer_exact b.B layout.bBound _ _
    (by rw [layout.bShape.1]; exact clamp_bound _ _ hk)
    (by rw [layout.bShape.2]; exact clamp_bound _ _ hn)

theorem c_offset_pointer (p : Parameters) (b : Buffers) (layout : Layout p.problem b)
    (strides : Strides p b) (hm : 0<p.problem.M) (hn : 0<p.problem.N)
    (tile : Nat) (i : CIndex) :
    (bytePointer b.C (cOffset p tile i)).toNat=
      b.C.pointer.toNat+b.C.dtype.bytes*(cOffset p tile i) := by
  have h := bytePointer_exact b.C layout.cBound
    (clamp p.problem.M (p.problem.row tile i.1.val))
    (clamp p.problem.N (p.problem.column tile i.2.1.val))
    (by rw [layout.cShape.1]; exact clamp_bound _ _ hm)
    (by rw [layout.cShape.2]; exact clamp_bound _ _ hn)
  simpa only [cOffset,strides.cm,strides.cn,Nat.mul_comm] using h

#axiomsClean scalar_bounds
#axiomsClean a_offset_pointer
#axiomsClean b_offset_pointer
#axiomsClean c_offset_pointer
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Pointer
