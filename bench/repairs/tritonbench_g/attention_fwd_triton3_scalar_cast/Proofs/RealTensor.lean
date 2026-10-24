import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.RawCorrect
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.TensorGeometry

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealTensor
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership TensorBinding
noncomputable section

/-- Ideal Real interpretation at the actual shared Tensor storage addresses.
This is separate from the raw FP32 word interpretation; it asserts no IEEE
representability, NaN-to-Real map, or equality with native arithmetic. -/
def logicalValue (heap : Nat → ℝ) (t : Torch.FP32.Tensor) (i : Nat) : ℝ :=
  if t.metadata.isNeg then -heap (t.storage.address (t.physicalIndex i))
  else heap (t.storage.address (t.physicalIndex i))

def initialState (A : FlatAlloc) (heap : Nat → ℝ) : BlockState where
  mem := fun rg off => MemCell.real (heap (4*A.addr rg off))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  undef := fun _ _ => 0

theorem six_inputs (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (a : Admission c A p r) (heap : Nat → ℝ)
    (role : Role) (i : Nat) (bound : i<(r role).tensor.metadata.numel) :
    (initialState A heap).readMem (region role)
      (offset (r role).tensor.metadata.shape (r role).strides i)=
      logicalValue heap (r role).tensor i := by
  rw [logicalValue,admitted_resolved c A p r a role]
  simp only [Bool.false_eq_true,if_false]
  rw [allocation_address c A p r a role i bound]
  rfl

def completedValue (A : FlatAlloc) (final : BlockState) (t : Torch.FP32.Tensor) (i : Nat) : ℝ :=
  logicalValue (fun byte => final.readMem A.flat (byte/4)) t i

#axiomsClean six_inputs
#specNonCircular logicalValue avoiding [RawKernel.whole_real,FP.Structural.exec,FP.Structural.run,Whole.result]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealTensor
