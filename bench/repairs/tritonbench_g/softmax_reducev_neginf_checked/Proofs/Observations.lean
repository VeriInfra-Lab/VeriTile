import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.TensorHost
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked
open VeriTile.Triton
noncomputable section

/-- Read the supplied Tensor's own storage identity and element offset. -/
def tensorValue {α : Type} (tensor : VeriTile.Torch.FP32.Tensor)
    (memory : Host.Memory α) (index : Nat) : α :=
  memory ⟨"torch-storage:" ++ toString tensor.storage.object⟩ (tensor.storageOffset+index)

@[simp] theorem tensorValue_object {α : Type} (object : Object) (memory : Host.Memory α) (i : Nat) :
    tensorValue object.tensor memory i=memory object.region (object.tensor.storageOffset+i) := rfl

/-- Little-endian byte observation of caller FP32 storage. -/
def memoryByte (memory : Host.Memory (BitVec 32)) (region : RegionName) (address : Nat) : BitVec 8 :=
  BitVec.ofNat 8 ((memory region (address/4)).toNat / 2^(8*(address%4)))

theorem byte_frame (c : TensorCall) (before after : Host.Memory (BitVec 32))
    (frame : ∀ r i,Exterior c r i → after r i=before r i) :
    ∀ r b,Exterior c r (b/4) → memoryByte after r b=memoryByte before r b := by
  intro r b hb
  simp only [memoryByte,frame r (b/4) hb]
#axiomsClean byte_frame
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked
