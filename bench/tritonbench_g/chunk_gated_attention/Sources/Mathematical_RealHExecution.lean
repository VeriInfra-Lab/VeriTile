import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHFinal

/-! Complete real execution of both typed H sources, at arbitrary symbolic
sizes and strides. All gate/seed/final switches and all iterations are covered.
The independent formula requires zero unspecified lanes; the source is not
silently changed to request zero padding. Finite-allocation safety, flat execution and public
IO packaging are proved in RealHSafety, RealHFlat and RealHIO. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHExecution
open VeriTile.Triton RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def footprint (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool) : WriteFootprint :=
  WriteFootprint.union (RealHLoop.footprint p l r) (RealHFinal.footprint p l r storeFinal)

structure Outputs (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed storeFinal : Bool) (v : BlockState) : Prop where
  history : ∀ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT → RealHStore.active p l e.val j.val →
    v.readMem r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)) =
      closed p l a gateK hasSeed c (keyIndex p l e.val) (valueIndex p l j.val)
  final : storeFinal = Bool.true → ∀ (e : Fin p.BK) (j : Fin p.BV), RealHStore.active p l e.val j.val →
    v.readMem r.final (stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val)) =
      closed p l a gateK hasSeed p.NT (keyIndex p l e.val) (valueIndex p l j.val)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHExecution
