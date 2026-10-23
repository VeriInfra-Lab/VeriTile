import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.Host

/-! Optional host lifecycle companion. Public Correct/FPEquiv concern successful
kernel execution; allocation/setup outcomes are recorded here. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.HostCorrect
open VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
noncomputable section

theorem wrapper_correct (x w out : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps)
    (red : Reduction) (p : Parameters) (u : Nat → ℝ)
    (rt : Runtime ℝ p entry live) (api : MemoryAPI p live)
    (b : Binding x w out eps p api.byteBase) (callerDevice : Nat) :
    Host.ResultSpec x p (mathematicalTensor p entry) rt callerDevice
      (Host.run x w out eps realNumerics red p u rt callerDevice) := by
  have same : logicalTensor realNumerics red p entry=mathematicalTensor p entry := by
    funext batch row k
    exact logicalValue_real red p.width p.eps _ _ k
  simpa only [same] using Host.complete x w out eps c realNumerics red p u rt api b callerDevice

abbrev wrapper_primitives := @Host.complete
abbrev rejected := @Host.rejected
#axiomsClean wrapper_correct
#axiomsClean wrapper_primitives
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.HostCorrect
