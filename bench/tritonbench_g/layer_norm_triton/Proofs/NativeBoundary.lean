import bench.tritonbench_g.layer_norm_triton.SourceCorrect

namespace VeriTile.Bench.TritonBenchG.LayerNormLock
open VeriTile.Triton

/-- The default PTX contains FMA at affine/variance/backward sites. Even finite
fp32 operands do not justify a context-free split-mul/add = FMA law. This is a
primitive graph counterexample, not a claim of a complete host-input kernel
counterexample. Bind an actual fused graph or select and verify nonfusion. -/
theorem no_unconditional_split_mul_add_fma :
    let a := FP.Value.ofBits .fp32 0x45800800 -- 4097
    let d := FP.Value.ofBits .fp32 0xcb801000 -- -(2^24 + 8192)
    FP.add .ieeeValue (FP.mul .ieeeValue a a) d = zero ∧
    FP.fma .ieeeValue a a d = one ∧
    FP.add .ieeeValue (FP.mul .ieeeValue a a) d ≠ FP.fma .ieeeValue a a d := by
  decide +kernel

#axiomsClean no_unconditional_split_mul_add_fma
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock
