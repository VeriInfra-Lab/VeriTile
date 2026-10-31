import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedLoopSafety
import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealFlatten
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedExecution
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedAccessSafety

/-! Complete repaired-source allocation safety. Only enabled memory roles
and actual local parent addresses are required. Every reached iteration is
proved safe using the already-proved original numerical invariant. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RebasedAccessSafety
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

abbrev Bounds := HSource.Bounds.Bounds

theorem source_safe (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool)
    (u : BlockState)
    (hfull : ∀ c, c < p.NT → RebasedAccess.Full p (RebasedInit.decoded p base u) c)
    (hkey : (RebasedInit.decoded p base u).key*p.BK+p.BK ≤ p.K)
    (hvalue : (RebasedInit.decoded p base u).value*p.BV+p.BV ≤ p.V)
    (hs : RealHStep.Separate r)
    (hinj : RealHLoop.HistoryNoCollision p (RebasedInit.decoded p base u))
    (bounds : RegionBounds) (hb : Bounds p (RebasedInit.decoded p base u) r gateK hasSeed storeFinal bounds) :
    (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel.TraceSafe bounds u := by
  exact HSource.Real.source_safe (RebasedProgram.parameters p base gateK hasSeed storeFinal swapped)
    r dot.dtype gate.dtype history.dtype seed final u hfull hkey hvalue hs hinj bounds hb

theorem source_flatten_ok (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) :
    (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel.FlattenOk := by
  exact HSource.Real.source_flatten_ok (RebasedProgram.parameters p base gateK hasSeed storeFinal swapped)
    r dot.dtype gate.dtype history.dtype seed final

#axiomsClean source_safe
#axiomsClean source_flatten_ok
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedSafety
