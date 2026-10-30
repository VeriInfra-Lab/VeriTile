import bench.tritonbench_g.matmul_triton1.Proofs.MatmulTriton1_Revision001_SharedContract_Contract
import bench.tritonbench_g.matmul_triton1.Proofs.FiniteIndex
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.TensorBinding
open VeriTile.Triton FP.Structural
open VeriTile.Torch
open SharedContract
noncomputable section
abbrev Tensor := VeriTile.Torch.FP32.Tensor
abbrev Laws := VeriTile.Torch.FP32.TensorLaws
abbrev Word := VeriTile.Torch.FP32.Word

/-- Ordinary local interpretation of the two source pointer arguments. Both
windows read the same physical memory, so overlapping inputs retain aliasing;
these equations do not describe output memory or any kernel execution. -/
structure Inputs {α : Type} (a b : Tensor) (la : Laws a) (lb : Laws b)
    (ca : a.metadata.isContiguous=Bool.true) (cb : b.metadata.isContiguous=Bool.true)
    (ra rb : RegionName) (read : Nat → Word) (decode : Word → α) (s : State α) : Prop where
  left : ∀ i, i<a.metadata.numel → s.mem ra i = Cell.mk .real
    (decode (read ((VeriTile.Torch.FP32.dataView a la ca).address i)))
  right : ∀ i, i<b.metadata.numel → s.mem rb i = Cell.mk .real
    (decode (read ((VeriTile.Torch.FP32.dataView b lb cb).address i)))

theorem logical_left {α : Type} [Inhabited α] (a b : Tensor) (la : Laws a) (lb : Laws b)
    (fixed default : Bool) (h : SharedContract.InputContract fixed default a.metadata b.metadata)
    (ra rb : RegionName) (read : Nat → Word) (decode : Word → α) (s : State α)
    (bound : Inputs a b la lb h.contiguous.1 h.contiguous.2 ra rb read decode s)
    (i : Nat) (hi : i<a.metadata.numel) :
    (s.mem ra i).read .real = decode (VeriTile.Torch.FP32.logicalValue read a la i) := by
  rw [bound.left i hi,VeriTile.Torch.FP32.direct_read_eq_logical read a la h.contiguous.1 h.noNeg.1 i hi]
  rfl

theorem logical_right {α : Type} [Inhabited α] (a b : Tensor) (la : Laws a) (lb : Laws b)
    (fixed default : Bool) (h : SharedContract.InputContract fixed default a.metadata b.metadata)
    (ra rb : RegionName) (read : Nat → Word) (decode : Word → α) (s : State α)
    (bound : Inputs a b la lb h.contiguous.1 h.contiguous.2 ra rb read decode s)
    (i : Nat) (hi : i<b.metadata.numel) :
    (s.mem rb i).read .real = decode (VeriTile.Torch.FP32.logicalValue read b lb i) := by
  rw [bound.right i hi,VeriTile.Torch.FP32.direct_read_eq_logical read b lb h.contiguous.2 h.noNeg.2 i hi]
  rfl

theorem span (t : Tensor) (laws : Laws t) (contiguous : t.metadata.isContiguous=Bool.true) :
    t.metadata.numel*4 ≤ 2^64 := by
  have h := (VeriTile.Torch.FP32.dataView t laws contiguous).spanFits
  change _+4*t.metadata.numel ≤ 2^64 at h
  omega

/-- The checked pointer, storage offset, alignment and four-byte word window
are bound to the same actual Torch input object. No cloned input supplies it. -/
theorem pointer_window (t : Tensor) (laws : Laws t)
    (contiguous : t.metadata.isContiguous=Bool.true)
    (aligned : 0<t.metadata.numel → t.dataPtr.toNat%4=0)
    (i : Nat) (hi : i<t.metadata.numel) :
    (t.dataPtr + FiniteIndex.word (i*4)).toNat =
      (VeriTile.Torch.FP32.dataView t laws contiguous).address i ∧
    (t.dataPtr + FiniteIndex.word (i*4)).toNat%4=0 ∧
    (t.dataPtr + FiniteIndex.word (i*4)).toNat+3 <
      (VeriTile.Torch.FP32.dataView t laws contiguous).base+t.metadata.numel*4 := by
  let view := VeriTile.Torch.FP32.dataView t laws contiguous
  have pointer := VeriTile.Torch.FP32.actual_pointer t laws contiguous (by omega)
  change FiniteIndex.word view.base = t.dataPtr at pointer
  rw [←pointer]
  have window : view.base+t.metadata.numel*4 ≤ 2^64 := by simpa [Nat.mul_comm] using view.spanFits
  have exact := FiniteIndex.accessed_bytes i t.metadata.numel view.base hi window
  have bytes : (FiniteIndex.word view.base+FiniteIndex.word (i*4)).toNat=view.base+i*4 := by
    simpa only [←BitVec.ofNat_mul] using exact.1
  rw [bytes]
  have align := VeriTile.Torch.FP32.dataView_aligned t laws contiguous aligned
  change view.base%4=0 at align
  refine ⟨by simp [view,VeriTile.Torch.FP32.DataView.address,Nat.mul_comm],?_,exact.2⟩
  omega

#axiomsClean logical_left
#axiomsClean logical_right
#axiomsClean pointer_window
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.TensorBinding
