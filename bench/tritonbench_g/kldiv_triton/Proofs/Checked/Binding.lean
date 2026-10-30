import bench.tritonbench_g.kldiv_triton.Proofs.Checked.Contract
import bench.tritonbench_g.kldiv_triton.Proofs.BackSafety
import bench.tritonbench_g.kldiv_triton.Proofs.Launch
import bench.tritonbench_g.kldiv_triton.Proofs.Grid

namespace VeriTile.Bench.Repairs.KldivTriton.Contiguous
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Host
noncomputable section
set_option autoImplicit false

/-- Ordinary metadata, storage and contiguous-view laws for the actual Tensor.
No kernel value, source result, reduction or numerical equality occurs here. -/
structure ViewBinding (t : Tensor) (v : View) : Prop where
  shape : v.shape = t.metadata.shape
  negative : v.negative = t.metadata.isNeg
  owner : v.storage.owner = t.metadata.device.index
  base : v.storage.baseByte = t.storageBase
  bytes : v.storage.bytes = t.storageBytes
  pointer : t.metadata.numel ≠ 0 →
    t.metadata.dataPtr.toNat = v.storage.baseByte + 4*v.storageOffset
  backed : v.backed

structure MatrixBinding (t : Tensor) (m : Matrix) : Prop extends ViewBinding t m.view where
  contiguous : t.metadata.isContiguous = Bool.true → Index.UnitColumns m ∧ m.injective
  metadata : L2NormBwd.Integer.Metadata m

theorem ViewBinding.numel {t : Tensor} {v : View} (h : ViewBinding t v) :
    v.numel = t.metadata.numel := by simp [View.numel,VeriTile.Torch.TensorMetadata.numel,h.shape]

theorem ViewBinding.separate {a b : Tensor} {u v : View}
    (hu : ViewBinding a u) (hv : ViewBinding b v) (h : StorageSeparate a b) :
    u.storage.separate v.storage := by
  rcases h with h | h
  · exact Or.inr (Or.inl (by simpa [hu.base,hu.bytes,hv.base] using h))
  · exact Or.inr (Or.inr (by simpa [hv.base,hv.bytes,hu.base] using h))

theorem MatrixBinding.rows_cols {a b : Tensor} {u v : Matrix}
    (hu : MatrixBinding a u) (hv : MatrixBinding b v) (h : b.metadata.shape = a.metadata.shape) :
    v.rows = u.rows ∧ v.cols = u.cols := by
  have shape := v.shape.symm.trans (hv.shape.trans (h.trans (hu.shape.symm.trans u.shape)))
  simpa using shape

theorem MatrixBinding.prepared {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    {t : Tensor} {m : Matrix} (h : MatrixBinding t m) (admit : TensorContract t)
    (s : World M) (live : s.live m.view.storage) : PreparedMatrix ops neg m m s s := by
  exact ⟨⟨rfl,h.negative.trans admit.noNeg,h.backed,live,rfl,fun _ _ => rfl,Or.inl rfl⟩,
    rfl,rfl,(h.contiguous admit.contiguous).1⟩

/-- All active source stores use the supplied descriptor's physical address,
including singleton/empty contiguous layouts with unused nonunit strides. -/
theorem MatrixBinding.store_address {t : Tensor} {m : Matrix} (h : MatrixBinding t m)
    (admit : TensorContract t) (r c : Nat) (hr : r < m.rows) (hc : c < m.cols) :
    m.key r c = m.view.storage.key (m.view.storageOffset+r*m.rowStride+c) := by
  simp only [Matrix.key,Matrix.index,(h.contiguous admit.contiguous).1.active r c hr hc]

/-- The checked actual data pointer proves alignment for every active word. -/
theorem MatrixBinding.aligned {t : Tensor} {m : Matrix} (h : MatrixBinding t m)
    (admit : TensorContract t) (r c : Nat) (hr : r < m.rows) (hc : c < m.cols) :
    (m.key r c).address % 4 = 0 := by
  have nonempty : t.metadata.numel ≠ 0 := by
    rw [←h.toViewBinding.numel,m.numel]
    exact Nat.ne_of_gt (Nat.mul_pos (by omega) (by omega))
  have align := admit.alignment nonempty
  rw [h.pointer nonempty] at align
  simp only [Matrix.key,Matrix.index,Allocation.key]
  omega

#axiomsClean MatrixBinding.store_address
#axiomsClean MatrixBinding.aligned
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Contiguous
