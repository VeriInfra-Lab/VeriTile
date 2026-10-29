import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.Host

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001

/-- Whole source-precision native graph, with only the eleven registered
local instruction equalities. Mathematical causal attention and its gradients
have an independent Real theorem; no exact FP-to-Real assertion is made. -/
theorem whole_native_fp32 (c : Shape) (actual reference : Native.Instructions)
    (owners : Native.Owners) (initial : Memory Native.Word) (scale : Native.Word)
    (numeric : Native.LocalConformance actual reference) :
    Native.Correct c reference owners initial
      (Native.run c (Native.ops actual) owners initial scale) scale :=
  Native.whole_native c actual reference owners initial scale numeric

#axiomsClean whole_native_fp32
#axiomsClean Host.whole_host
#specNonCircular Native.LocalConformance avoiding [Native.run,Native.Correct,whole_native_fp32,Host.whole_host]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.TritonAttention.Whole001
