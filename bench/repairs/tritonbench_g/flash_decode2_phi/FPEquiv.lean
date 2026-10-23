import bench.repairs.tritonbench_g.flash_decode2_phi.Contract
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.WholeNative

namespace VeriTile.Bench.Repairs.FlashDecode2Phi.FPEquiv
open VeriTile.Triton FP.Structural
open FlashDecode2 FlashDecode2.TensorBinding
noncomputable section

theorem actual_source (c : Call) :
    FlashDecode2.kernel c.val.masked (Checked.heads c.val) (Checked.capacity c.val)
      (Checked.width c.val) c.val.blockSeq (Checked.block c.val) =
    FlashDecode2.phi (Checked.heads c.val) (Checked.capacity c.val)
      (Checked.width c.val) c.val.blockSeq (Checked.block c.val) := by
  simp only [FlashDecode2.kernel,c.property]; rfl

def invoke (p : Numerical.ScalarOps) (schedule : Native.Schedule) (c : Call)
    (t : Tensors c.val allocations) (memory : Physical.Bytes) :=
  WholeNative.invoke p schedule c.val t memory

theorem tensor (actual reference : Numerical.ScalarOps) (laws : Numerical.ScalarLaws actual reference)
    (schedule : Native.Schedule) (c : Call) (t : Tensors c.val allocations) (memory : Physical.Bytes)
    (h : InputContract c (lengthValues t memory)) :
    ∃ result,invoke actual schedule c t memory = .ok result ∧
      WholeNative.Guarantee actual reference schedule c.val t memory h result :=
  WholeNative.tensor actual reference laws schedule c.val t memory h

theorem empty (p : Numerical.ScalarOps) (schedule : Native.Schedule) (c : Call)
    (t : Tensors c.val allocations) (memory : Physical.Bytes)
    (h : InputContract c (lengthValues t memory))
    (empty : Checked.batches c.val = 0 ∨ Checked.heads c.val = 0) :
    invoke p schedule c t memory = .ok ⟨none,t.output.tensor,memory,t.callerDevice,none⟩ :=
  WholeNative.empty p schedule c.val t memory h empty

#axiomsClean actual_source
#axiomsClean tensor
#axiomsClean empty
end
end VeriTile.Bench.Repairs.FlashDecode2Phi.FPEquiv
