import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeRealExecution
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedCumAccess

/-! Complete repaired cumulative execution, including flattened launch
initialization and exhausted autotune batches. The actual loaded/stored prefix
and tagged-cell frame use the existing local triangular-dot proof. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumExecution
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG
open ChunkGatedAttention
open RealCumProgram
open ChunkCumsumVector (lowerTriTile)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def linear (base : Nat) (u : BlockState) := u.pids 0+base

def decoded (p : Layout) (base : Nat) (u : BlockState) : Launch :=
  ⟨linear base u % (p.S/p.BS), linear base u / ((p.S/p.BS)*(p.T/p.BT))⟩

def chunk (p : Layout) (base : Nat) (u : BlockState) :=
  (linear base u / (p.S/p.BS)) % (p.T/p.BT)

def q (p : Layout) (base : Nat) (u : BlockState) :=
  RebasedCumAccess.tileLayout p (decoded p base u) (chunk p base u)

def enabled (p : Layout) (heads base : Nat) (u : BlockState) :=
  linear base u < heads*(p.T/p.BT)*(p.S/p.BS)
instance (p : Layout) (heads base : Nat) (u : BlockState) : Decidable (enabled p heads base u) :=
  inferInstanceAs (Decidable (_ < _))

def footprint (p : Layout) (heads base : Nat) (o : RegionName) (u : BlockState) : WriteFootprint :=
  fun cell => enabled p heads base u ∧
    RealCumExecution.footprint (q p base u) RebasedCumAccess.localLaunch 0 o cell

theorem executes (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool)
    (dtype out_dtype : FloatDType) (u : BlockState)
    (hinj : enabled p heads base u →
      RealCumExecution.NoCollision (q p base u) RebasedCumAccess.localLaunch 0) :
    ∃ v, exec (RebasedCumProgram.source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel u = some v ∧
      (enabled p heads base u → ∀ i : TileIndex [p.BT,p.BS],
        v.readMem o (RealCumExecution.offset (q p base u) RebasedCumAccess.localLaunch 0 i) =
          RealCumExecution.expected (q p base u) RebasedCumAccess.localLaunch 0 (u.readMem s) i) ∧
      BlockState.WriteWithin (footprint p heads base o u) u v := by
  have h := CumulativeSource.Real.executes (RebasedCumProgram.parameters p heads base swapped)
    dtype out_dtype s o u
    (by simpa [RebasedCumProgram.parameters,CumulativeSource.Real.enabled,
      CumulativeSource.Real.linearIndex,CumulativeSource.Real.tileLayout,CumulativeSource.Real.tileBase,
      CumulativeSource.Real.headIndex,CumulativeSource.Real.chunkIndex,CumulativeSource.Real.featureIndex,
      CumulativeSource.Real.localLaunch,enabled,linear,q,decoded,chunk,
      RebasedCumAccess.tileLayout,RebasedCumAccess.tileBase,RebasedCumAccess.localLaunch] using hinj)
  simpa [RebasedCumProgram.source,CumulativeSource.kernel,RebasedCumProgram.parameters,
    CumulativeSource.Real.enabled,CumulativeSource.Real.linearIndex,CumulativeSource.Real.tileLayout,
    CumulativeSource.Real.tileBase,CumulativeSource.Real.headIndex,CumulativeSource.Real.chunkIndex,
    CumulativeSource.Real.featureIndex,CumulativeSource.Real.localLaunch,CumulativeSource.Real.footprint,
    enabled,linear,q,decoded,chunk,footprint,RebasedCumAccess.tileLayout,RebasedCumAccess.tileBase,
    RebasedCumAccess.localLaunch] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumExecution
