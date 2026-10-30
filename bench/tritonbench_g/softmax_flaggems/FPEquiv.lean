import bench.tritonbench_g.softmax_flaggems.Proofs.SourcePair

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.FPEquiv
open VeriTile Triton FP.ProfiledRewrite SourcePair
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open scoped VeriTile.Spec
noncomputable section

/-- The actual repaired finite-seed forward source and its pre-existing
address candidate, including inner/noninner and one/multiple-tile branches.
Only natural pointer multiplication changes. Full native IO observations,
all words, aliases and consumer reductions remain in Misc/NativeCorrect.lean. -/
specification source_forward (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) :
    forwardOriginal X Y d c ≡[assumptions] forwardAddressCommuted X Y d c :=
  SourcePair.forward_equivalent X Y d c

/-- Both complete backward source branches and their existing address
candidates, with all FP32 arithmetic, casts and store order preserved. -/
specification source_backward (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) :
    backwardOriginal P G Y d c ≡[assumptions] backwardAddressCommuted P G Y d c :=
  SourcePair.backward_equivalent P G Y d c

#print_fp_assumptions source_forward
#print_fp_assumptions source_backward
#axiomsClean source_forward
#axiomsClean source_backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.FPEquiv
