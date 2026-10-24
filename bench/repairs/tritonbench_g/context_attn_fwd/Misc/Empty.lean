import bench.repairs.tritonbench_g.context_attn_fwd.Misc.Public

namespace VeriTile.Bench.Repairs.ContextAttnFwd
open VeriTile.Triton FP.Structural NativeGraph Returned
noncomputable section
local instance (c : Host.Call) : Decidable (Needed c) := Classical.propDecidable _

/-- The executable skip branch needs no allocation separation, geometry,
floating-input, or numerical premise. Empty views may alias the supplied Out.
Every numerical field is unevaluated; arbitrary instruction operations give
exactly the original bytes and graph snapshot. -/
theorem empty_return (c : Host.Call) (B : Native.Bytes extent) (ops : ScalarOps)
    (ctx : Context) (before : Physical.Memory) (empty : ¬Needed c) :
    ∃ result,invoke c B ops ctx before=some result ∧ result.bytes=before ∧
      result.returned=none ∧ result.suppliedOutput=ctx.suppliedOutput ∧
      result.currentDevice=ctx.callerDevice ∧ result.launchDevice=none ∧
      result.graph=Snapshot.initial c B before := by
  have hb := Physical.empty_stores c B ops (Snapshot.initial c B before) before (empty_lengths c empty)
  refine ⟨⟨Snapshot.initial c B before,before,none,ctx.suppliedOutput,ctx.callerDevice,none⟩,?_,rfl,rfl,rfl,rfl,rfl,rfl⟩
  simp only [invoke,order,if_neg empty,FPSuccess.runGrid]
  change some _=some _
  rw [hb]

#print axioms empty_return
end
end VeriTile.Bench.Repairs.ContextAttnFwd
