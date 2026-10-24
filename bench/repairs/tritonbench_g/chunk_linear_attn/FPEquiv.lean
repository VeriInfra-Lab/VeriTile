import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.ProgramPair

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.FPEquiv
open VeriTile Triton
open scoped VeriTile.Spec
noncomputable section

abbrev original := ProgramPair.original
abbrev hoisted := ProgramPair.hoisted

/-- Complete four-stage original versus actual head-address CSE candidate,
followed by the unchanged finite FP32 dV sum. All raw FP32 words, the runtime
scale word, every original native scalar/FMA profile, optional endpoints,
all permitted per-stage grid permutations, all nine observable channels,
success, source safety and both complete exterior frames are retained.
Only exact integer address evaluation is shared; no numerical rule is used. -/
specification whole_equiv (profile : NativeTerms.FP32Profile) (c : KernelInterface.Config)
    (scale : BitVec 32) (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c.call work) :
    original profile c scale work ≡[[]] hoisted profile c scale work :=
  Spec.FloatingPoint.ofStructural rfl rfl (ProgramPair.equivalent profile c scale work legal)

#axiomsClean whole_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.FPEquiv
