import bench.tritonbench_g.decay_cumsum.Proofs.SourcePair

namespace VeriTile.Bench.Repairs.DecayCumsum.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

abbrev original := SourcePair.original
abbrev indexCommuted := SourcePair.indexCommuted
abbrev assumptions := SourcePair.assumptions

/-- Every complete FP32 forward, preparation and backward main/tail template
is equivalent to its existing mask-index candidate. Only exact natural index
multiplication changes; all numerical instructions and failures are retained.
The unrestricted compiled native IO theorem remains in Misc/NativeCorrect.lean. -/
specification decay_cumsum_source_equiv (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    original kind h p scale ≡[assumptions] indexCommuted kind h p scale :=
  SourcePair.equivalent kind h p scale

#print_fp_assumptions decay_cumsum_source_equiv
#axiomsClean decay_cumsum_source_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DecayCumsum.FPEquiv
