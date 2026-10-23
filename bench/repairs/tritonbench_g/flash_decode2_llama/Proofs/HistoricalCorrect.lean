import bench.repairs.tritonbench_g.flash_decode2_llama.Contract
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.WholeReal

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.Historical.Correct
open VeriTile.Triton FP.Structural
open FlashDecode2 FlashDecode2.TensorBinding
noncomputable section

theorem actual_source (c : Call) :
    FlashDecode2.kernel c.val.masked (Checked.heads c.val) (Checked.capacity c.val)
      (Checked.width c.val) c.val.blockSeq (Checked.block c.val) =
    FlashDecode2.llama (Checked.heads c.val) (Checked.capacity c.val)
      (Checked.width c.val) c.val.blockSeq (Checked.block c.val) := by
  simp only [FlashDecode2.kernel,c.property]; rfl

def invoke (c : Call) (t : Tensors c.val allocations) (memory : Physical.Bytes) (heap : WholeReal.Heap) :=
  WholeReal.invoke c.val t memory heap

theorem tensor (c : Call) (t : Tensors c.val allocations) (memory : Physical.Bytes) (heap : WholeReal.Heap)
    (h : InputContract c (lengthValues t memory)) :
    ∃ result,invoke c t memory heap = .ok result ∧ WholeReal.Guarantee c.val t memory heap result :=
  WholeReal.tensor c.val t memory heap h

theorem empty (c : Call) (t : Tensors c.val allocations) (memory : Physical.Bytes) (heap : WholeReal.Heap)
    (h : InputContract c (lengthValues t memory))
    (empty : Checked.batches c.val = 0 ∨ Checked.heads c.val = 0) :
    invoke c t memory heap = .ok ⟨none,t.output.tensor,fun r i => some (heap r i),t.callerDevice,none⟩ :=
  WholeReal.empty c.val t memory heap h empty

#axiomsClean actual_source
#axiomsClean tensor
#axiomsClean empty
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.Historical.Correct
