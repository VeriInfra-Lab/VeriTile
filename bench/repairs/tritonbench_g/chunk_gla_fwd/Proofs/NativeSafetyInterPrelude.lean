import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyBasics

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPInter
open FPOutput (DotPrimitive withDot)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

structure InterCapacity (c : ProducerGeometry.Shape) (s : State α)
    (bounds : RegionName → Nat) : Prop where
  head : s.pids 2 < c.heads
  query : c.heads*c.T*c.K ≤ bounds "q"
  key : c.heads*c.T*c.K ≤ bounds "k"
  gate : c.heads*c.T*c.K ≤ bounds "g"
  output : c.heads*c.T*c.BT ≤ bounds "attention"

theorem inter_prologue (M : Algebra α) (D : DotPrimitive α)
    (c : ProducerGeometry.Shape) (scale : ℝ) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (engine (withDot M D)) bounds (outer c scale).before s := by
  apply pure_trace
  simp [outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,
    ProducerGeometry.layout,ProducerGeometry.regions,ComputeKernel.surfaceBody,
    Source.conditional?,Source.prefixBefore]
  all_goals try simp only [Op.tritonDiv,Op.tritonMod]
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem inter_entry (M : Algebra α) (D : DotPrimitive α)
    (c : ProducerGeometry.Shape) (scale : ℝ) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale).before s := by
  apply pure_trace
  simp [parts,FPInter.inner,outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,
    ProducerGeometry.layout,ProducerGeometry.regions,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore]
  all_goals try simp only [Op.tritonDiv,Op.tritonMod]
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

#axiomsClean inter_prologue
#axiomsClean inter_entry

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
