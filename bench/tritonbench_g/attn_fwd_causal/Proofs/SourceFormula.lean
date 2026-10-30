import bench.tritonbench_g.attn_fwd_causal.IO

namespace VeriTile.Bench.Repairs.AttnFwdCausal.Reference
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- Independent fixed-instruction two-fold native formula. Original cast,
reduction ownership, TF32/FP16 products and FMA order remain unchanged. -/
def native (instructions : Native.Instructions) (p : Parameters) (vals : IO.Inputs Native.Word p)
    (j : TileIndex [p.batches,p.heads,p.length,128]) : Native.Word :=
  (NativeReference.attention p.length j.2.2.1.val
    (nativeInputs p vals j.1.val j.2.1.val) j.2.2.2.1).eval instructions.ops

#specNonCircular native avoiding [Whole.answer,Source.accumulated,Source.update,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Reference
