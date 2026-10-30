import bench.tritonbench_g.kldiv_compute.Proofs.Checked.BindingIndex
import bench.tritonbench_g.swiglu_triton.Proofs.Memory
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.KldivCompute.Kernel
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Contiguous
export VeriTile.Torch.FP32 (Tensor TensorLaws)
noncomputable section
set_option autoImplicit false

/-- Independent pointwise KL term; no reduction or zero-target extension. -/
def mathematical (x y : ℝ) : ℝ := x * Real.log (x/y)

def allocation (t : Tensor) : Allocation :=
  ⟨t.storage.object,t.metadata.device.index,t.storage.base,4*t.storage.elements,t.storage.spanFits⟩

def key (t : Tensor) (i : Nat) : Key := (allocation t).key (t.physicalIndex i)
def directKey (t : Tensor) (i : Nat) : Key := (allocation t).key (t.storageOffset+i)

/-- Logical Tensor contents use its physical coordinate map and negative bit.
The same definition works over arbitrary Real cells and native FP32 words. -/
def logical {M α β : Type} (ops : Ops M α β) (negate : α → α)
    (memory : M) (t : Tensor) (i : Nat) : α :=
  if t.metadata.isNeg then negate (ops.read memory (key t i)) else ops.read memory (key t i)

/-- Exactly the selected input checks; the remaining facts are the existing
fresh output allocation and ordinary Tensor storage bindings. Input aliases,
nonzero offsets, arbitrary rank/scalars/empties remain admitted. -/
structure Domain (x y output : Tensor) : Prop where
  admitted : InputContract x.metadata y.metadata
  xLaws : TensorLaws x
  yLaws : TensorLaws y
  outputLaws : TensorLaws output
  outputShape : output.metadata.shape = x.metadata.shape
  outputDevice : output.metadata.device = x.metadata.device
  outputContiguous : output.metadata.isContiguous = Bool.true
  outputNoNeg : output.metadata.isNeg = Bool.false
  outputAligned : 0 < output.metadata.numel → output.dataPtr.toNat % 4 = 0
  freshX : (allocation x).separate (allocation output)
  freshY : (allocation y).separate (allocation output)

abbrev python_checks_iff_preconditions := Contiguous.checkInputs_iff

#specNonCircular mathematical avoiding [KernelDSL.kernel]
#specNonCircular Domain avoiding [KernelDSL.kernel,mathematical]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Kernel
