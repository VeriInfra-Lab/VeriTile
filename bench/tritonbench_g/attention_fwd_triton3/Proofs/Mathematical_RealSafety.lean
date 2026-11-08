import bench.tritonbench_g.attention_fwd_triton3.Proofs.Mathematical_RealSafetyPrologue
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Mathematical_RealSafetyLoop

/-! Shared mathematical address bounds for isolated Real proof fragments. Complete actual-source trace safety is proved in Proofs/SourceSafety.lean. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSafety
open VeriTile.Triton RealAccess RealPrologue
noncomputable section

structure Bounds (BM BN BD finish : Nat) (p : Parameters) (s : BlockState) (bounds : RegionBounds) : Prop
    extends RealSafetyPrologue.Bounds BM BN BD p s bounds where
  keys : ∀ j, j < RealLoop.count (loopParameters BM BN BD p s) →
    BlockBounds bounds (RealLoop.keyPointer (loopParameters BM BN BD p s) j) (RealAccumulation.checks p.evenN)
  values : ∀ j, j < RealLoop.count (loopParameters BM BN BD p s) →
    BlockBounds bounds (RealLoop.valuePointer (loopParameters BM BN BD p s) j) (RealAccumulation.checks p.evenN)
  writeCenter : ∀ i, RealExecution.rowAddress BM p s i < bounds p.layout.M
  writeDenominator : finish = 0 → ∀ i, RealExecution.rowAddress BM p s i < bounds p.layout.L
  writeOutput : ∀ i, RealExecution.outputAddress BM BD p s i < bounds p.layout.Out


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSafety
