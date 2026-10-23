import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.Binding

namespace VeriTile.Bench.Repairs.RotaryTransformOps.HistoricalPublic
open Checked Binding

/- Complete FP32 words, actual returned Tensor, every output byte, and physical frame. -/
theorem rotary_transform_ops_fp32 {M : Type} (api : API M Native.Word Physical.Byte) (codec : Codec api)
    (c : Call) (out : Output c) (callerDevice : Nat) (p : Native.Instructions) (m : M)
    (accepted : check c (mirror api c out m)=Bool.true)
    (pointers : pointerCheck c out (mirror api c out m)=Bool.true) :
    ∃ result,invoke api c out callerDevice p m=some result ∧ result.tensor=out.tensor ∧
      result.deviceRestored=callerDevice ∧
      (c.host.shape.inplace=Bool.true → result.tensor=c.input) ∧
      (∀ b r h d,b<c.host.shape.batches → r<(configuration c (mirror api c out m)).length b →
        h<c.host.shape.heads → d<c.host.shape.width →
        api.read result.memory (key result.tensor (c.host.shape.address
          ((configuration c (mirror api c out m)).start b+r) h d))=
            expected c p (mirror api c out m) b r h d) ∧
      (∀ b r h d i,b<c.host.shape.batches → r<(configuration c (mirror api c out m)).length b →
        h<c.host.shape.heads → d<c.host.shape.width →
        api.observe result.memory result.tensor.device (result.tensor.base+4*c.host.shape.address
          ((configuration c (mirror api c out m)).start b+r) h d+i.val)=
            Physical.encode (expected c p (mirror api c out m) b r h d) i) ∧
      (∀ owner address,Outside c out owner address →
        api.observe result.memory owner address=api.observe m owner address) :=
  Binding.complete api codec c out callerDevice p m accepted pointers

#axiomsClean rotary_transform_ops_fp32
end VeriTile.Bench.Repairs.RotaryTransformOps.HistoricalPublic
