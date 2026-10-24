import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.NativeResults

/-! Complete FP32 source-to-native observation. Fixed stored words, per-consumer
reductions and scalar lowering remain explicit; all words and source/output
aliases are admitted. This is an independent native value graph, not a second
kernel or a Real equality. Numerical transformation assumptions: none. -/
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.FPEquiv
open VeriTile Triton FP.Structural Native Interface
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section

theorem kernel_forward (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B) :
    (forwardNativeIO ops scalar R S X Y d c).Implements
      (forwardNativeFormula ops scalar R S X Y d c) :=
  forward_native ops scalar R S X Y d c hN hK hB hC hone

theorem kernel_backward (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B) :
    (backwardNativeIO ops scalar R S P G Y d c).Implements
      (backwardNativeFormula ops scalar R S P G Y d c) :=
  backward_native ops scalar R S P G Y d c hN hK hB hC hone

#axiomsClean kernel_forward
#axiomsClean kernel_backward
#stmtSurfaceSubset kernel_forward ⊆
  [NativeProfile.Operations,Word,Reductions,Native.Schedule,RegionName,Dimensions,Dimensions.N,
   Dimensions.K,LaunchConfig,LaunchConfig.B,LaunchConfig.C,LaunchConfig.one,oneTile,
   forwardNativeIO,forwardNativeFormula,AddressedProgramIO.Implements,
   Native.Term,Native.Consumer,Scalar.Term,SoftmaxFlaggemsFiniteSeed.Native.instInhabitedTerm]
#stmtSurfaceSubset kernel_backward ⊆
  [NativeProfile.Operations,Word,Reductions,Native.Schedule,RegionName,Dimensions,Dimensions.N,
   Dimensions.K,LaunchConfig,LaunchConfig.B,LaunchConfig.C,LaunchConfig.one,oneTile,
   backwardNativeIO,backwardNativeFormula,AddressedProgramIO.Implements,
   Native.Term,Native.Consumer,Scalar.Term,SoftmaxFlaggemsFiniteSeed.Native.instInhabitedTerm]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.FPEquiv
