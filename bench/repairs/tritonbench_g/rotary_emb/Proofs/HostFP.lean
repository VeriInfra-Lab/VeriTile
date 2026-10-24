import bench.repairs.tritonbench_g.rotary_emb.Proofs.PublicFPEquiv
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Binding

namespace VeriTile.Bench.Repairs.RotaryEmb
open Checked Binding

/-- Complete caller-addressed raw FP32 result and byte frame. Numerical ports
are scalar mul/neg/fma only; source execution and outputs are proved. -/
theorem rotary_emb_fp32 {M : Type} (api : API M Native.Word Physical.Byte) (codec : Codec api)
    (p : Native.Instructions) (c : Call) (accepted : check c=Bool.true) (callerDevice : Nat) (m : M) :
    ∃ r,invoke api p c callerDevice m=some r ∧ r.value=none ∧ r.tensor=c.tensor ∧
      r.deviceRestored=callerDevice ∧
      (∀ which n h d,n<c.shape.tokens → h<c.shape.heads which → d<c.shape.width →
        api.read r.memory (tensorKey (r.tensor (role which)) (c.shape.address which n h d))=
          expected api p c m which n h d) ∧
      (∀ which n h d i,n<c.shape.tokens → h<c.shape.heads which → d<c.shape.width →
        api.observe r.memory (r.tensor (role which)).device
          ((r.tensor (role which)).base+4*c.shape.address which n h d+i.val)=
          Physical.encode (expected api p c m which n h d) i) ∧
      (∀ owner address,Outside c owner address → api.observe r.memory owner address=api.observe m owner address) :=
  Binding.complete api codec p c accepted callerDevice m

#axiomsClean rotary_emb_fp32
end VeriTile.Bench.Repairs.RotaryEmb
