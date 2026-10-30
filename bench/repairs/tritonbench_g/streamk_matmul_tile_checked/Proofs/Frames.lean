import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.KernelMemory
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.Tensor

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.KernelMemory
open Whole001 Source Checked
noncomputable section

theorem values_frame (v : ByteExecution.Layout p) (s : Source.State α p exterior)
    (before : TensorBinding.ValueRead α) (address : Nat)
    (hc : ByteMemory.Outside (ByteExecution.cView v) address)
    (hs : ByteMemory.Outside (ByteExecution.sView v) address) :
    ValueMemory.rendered before v s address=before address := by
  unfold ValueMemory.rendered
  change address<v.outputBase ∨ v.outputBase+4*(p.h.m*p.h.n)≤address at hc
  rw [ValueMemory.frame _ v.outputBase s.output address hc]
  split
  next active =>
    apply ValueMemory.frame
    simpa only [ByteMemory.Outside,ByteExecution.sView,active,ite_true] using hs
  next => rfl

theorem bytes_frame (v : ByteExecution.Layout p) (s : Source.State TensorBinding.Word p exterior)
    (before : ByteMemory.Read) (address : Nat)
    (hc : ByteMemory.Outside (ByteExecution.cView v) address)
    (hs : ByteMemory.Outside (ByteExecution.sView v) address) :
    ByteExecution.rendered before v s address=before address := by
  unfold ByteExecution.rendered
  rw [ByteMemory.render_frame _ _ _ _ hc,ByteMemory.render_frame _ _ _ _ hs]

theorem values_input_frame (a b : VeriTile.Torch.FP32.Tensor) (v : ByteExecution.Layout p)
    (out : VeriTile.Torch.FP32.Tensor) (buffers : TensorModel.Buffers a b p v out)
    (s : Source.State α p exterior) (before : TensorBinding.ValueRead α) (address : Nat)
    (input : (a.storage.base≤address ∧ address<a.storage.base+4*a.storage.elements) ∨
      (b.storage.base≤address ∧ address<b.storage.base+4*b.storage.elements)) :
    ValueMemory.rendered before v s address=before address := by
  apply values_frame
  · rcases input with hi|hi
    · exact TensorModel.outside_storage _ a buffers.outputA address hi
    · exact TensorModel.outside_storage _ b buffers.outputB address hi
  · rcases input with hi|hi
    · exact TensorModel.outside_storage _ a buffers.scratchA address hi
    · exact TensorModel.outside_storage _ b buffers.scratchB address hi

theorem bytes_input_frame (a b : VeriTile.Torch.FP32.Tensor) (v : ByteExecution.Layout p)
    (out : VeriTile.Torch.FP32.Tensor) (buffers : TensorModel.Buffers a b p v out)
    (s : Source.State TensorBinding.Word p exterior) (before : ByteMemory.Read) (address : Nat)
    (input : (a.storage.base≤address ∧ address<a.storage.base+4*a.storage.elements) ∨
      (b.storage.base≤address ∧ address<b.storage.base+4*b.storage.elements)) :
    ByteExecution.rendered before v s address=before address := by
  apply bytes_frame
  · rcases input with hi|hi
    · exact TensorModel.outside_storage _ a buffers.outputA address hi
    · exact TensorModel.outside_storage _ b buffers.outputB address hi
  · rcases input with hi|hi
    · exact TensorModel.outside_storage _ a buffers.scratchA address hi
    · exact TensorModel.outside_storage _ b buffers.scratchB address hi

#axiomsClean values_frame
#axiomsClean bytes_frame
#axiomsClean values_input_frame
#axiomsClean bytes_input_frame
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.KernelMemory
