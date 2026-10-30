import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.LogicalReader
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.Correct
open VeriTile.Triton
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO
noncomputable section

/-- Inclusive time prefix at each feature of one complete feature-tile
invocation. All nine configurations, the dynamic carry loop, actual strides,
active tails, source access safety and complete exterior frame are retained. -/
specification memory_correct (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (chosen : c ∈ configurations) (p : Geometry.Program h) :
    Spec.Real (PublicRealIO.prefixLogicalIO h c p ⊨ fun xs cell =>
      ∑ time : Fin h.T, if time ≤ cell.1 then xs time cell.2 else 0) :=
  PublicRealIO.logical_implements h layout c chosen p

#axiomsClean memory_correct
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.Correct
