import bench.tritonbench_g.kldiv_ops.Proofs.Checked.Contract
import bench.tritonbench_g.kldiv_triton.Proofs.Checked.Binding

namespace VeriTile.Bench.Repairs.KldivOps.Contiguous
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
set_option autoImplicit false

/-- Ordinary active row-major contiguity; no restriction on unused strides. -/
def Dense (v : View) : Prop :=
  ∀ i, i < v.numel → v.key i = v.storage.key (v.storageOffset+i)

structure Binding (t : Tensor) (v : View) : Prop extends KldivTriton.Contiguous.ViewBinding t v where
  contiguous : t.metadata.isContiguous = Bool.true → Dense v

theorem Dense.unique {v : View} (h : Dense v) (i j : Nat)
    (hi : i < v.numel) (hj : j < v.numel) (eq : v.key i = v.key j) : i = j := by
  rw [h i hi,h j hj] at eq
  have addresses := congrArg Key.address eq
  simp only [Allocation.key] at addresses
  omega

theorem Dense.disjoint {v : View} (h : Dense v) (i j : Nat)
    (hi : i < v.numel) (hj : j < v.numel) (different : i ≠ j) : (v.key i).disjoint (v.key j) := by
  rw [h i hi,h j hj]
  apply Allocation.indices_disjoint
  omega

theorem Binding.raw {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    {t : Tensor} {v : View} (h : Binding t v) (admit : TensorContract t)
    (memory : M) (i : Nat) (hi : i < v.numel) :
    ops.read memory (v.storage.key (v.storageOffset+i)) = logical ops neg memory v i := by
  rw [←h.contiguous admit.contiguous i hi]
  simp [logical,h.negative,admit.noNeg,Negation.encode]

#axiomsClean Dense.unique
#axiomsClean Binding.raw
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous
