import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealExecution
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedProgram
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedInit
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHExecution

/-! Complete repaired typed H source, all chunks and all source storage/gate/
seed/final variants. The unchanged independent closed form and output/frame
interfaces are imported. Full tiles follow from staging, not an input-shape
restriction; no oracle-value premise remains. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedExecution
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
abbrev footprint := RealHExecution.footprint
abbrev Outputs := RealHExecution.Outputs

theorem executes (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) (a : Inputs) (u : BlockState)
    (hfull : ∀ c, c < p.NT → RebasedAccess.Full p (RebasedInit.decoded p base u) c)
    (hkey : (RebasedInit.decoded p base u).key*p.BK+p.BK ≤ p.K)
    (hvalue : (RebasedInit.decoded p base u).value*p.BV+p.BV ≤ p.V)
    (hs : RealHStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final)
    (hhi : RealHLoop.HistoryNoCollision p (RebasedInit.decoded p base u))
    (hfi : storeFinal = Bool.true → RealHStore.NoCollision p (RebasedInit.decoded p base u) (RealHFinal.base p (RebasedInit.decoded p base u)) p.V 1)
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hg : ∀ off, u.readMem r.gate off = a.gate off)
    (hseed : ∀ off, u.readMem r.seed off = a.seed off) :
    ∃ v, exec (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel u = some v ∧
      Outputs p (RebasedInit.decoded p base u) r a gateK hasSeed storeFinal v ∧
      BlockState.WriteWithin (footprint p (RebasedInit.decoded p base u) r storeFinal) u v := by
  exact HSource.Real.executes (RebasedProgram.parameters p base gateK hasSeed storeFinal swapped)
    r dot.dtype gate.dtype history.dtype seed final a u hfull hkey hvalue hs hfsep hhi hfi hk hv hg hseed

#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedExecution
