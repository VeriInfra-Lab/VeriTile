import bench.repairs.tritonbench_g.rope_transform.Proofs.Public

namespace VeriTile.Bench.Repairs.RopeTransform
open VeriTile Triton FP.Structural
/-- Guarded whole-source FP32 words and physical bytes. Only local instruction
semantics, shape-only compiler selection, and ordinary tensor/memory API laws
are parameters; no successful execution or whole-output equality is a premise. -/
theorem rope_forward_fp32 {M : Type}
    (ops : VeriTile.Bench.Repairs.SwigluTriton.Memory.Ops M Native.Word Physical.Byte)
    (codec : Physical.Codec ops) (I : Native.Instructions) (S : Native.Schedule)
    (r : Request) (call : Host.Preparation Native.Term r)
    (memory : RegionName → Nat → Native.Word) (before prepared : M)
    (binding : Physical.Binding ops I S r call memory before prepared)
    (entry : call.before=Native.entry memory) (accepted : Checked.check r call=Bool.true) :
    Checked.FP32Outcome ops I S r call memory before prepared :=
  Physical.complete ops codec I S r call memory before prepared binding entry accepted
#axiomsClean rope_forward_fp32
end VeriTile.Bench.Repairs.RopeTransform
