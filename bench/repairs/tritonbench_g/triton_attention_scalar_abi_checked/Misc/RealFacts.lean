import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Correct

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Correct
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

/-- The independent gradient formulas are the full simultaneous Q/K/V VJP. -/
abbrev attention_vjp := @Math.attention_vjp

#axiomsClean attention_vjp
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Correct
