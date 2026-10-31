import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeRealProgram
import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeErasure
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealCumProgram

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumProgram
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumProgram

def parameters (p : Layout) (heads base : Nat) (swapped : Bool) : CumulativeSource.Parameters :=
  ⟨p.headStride,p.timeStride,p.channelStride,p.T,p.S,p.BT,p.BS,heads,base,swapped⟩

def source (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool)
    (dtype out_dtype : FloatDType) : ComputeKernel :=
  (if swapped then NativeSource.cumIndexFirst else NativeSource.cumOriginal)
    dtype out_dtype s o p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS heads base

/-- A checked mathematical view of the complete repaired source. -/
def body (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool) : List Stmt :=
  (CumulativeSource.Real.algorithm (parameters p heads base swapped) s o).body

theorem source_body (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool)
    (dtype out_dtype : FloatDType) :
    (source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel.body=
      body p s o heads base swapped := by
  change (CumulativeSource.kernel (parameters p heads base swapped) dtype out_dtype s o).eraseDType.toAlgKernel.body=_
  rw [CumulativeSource.erase_connection,CumulativeSource.erased_algorithm]
  rfl

#axiomsClean source_body
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumProgram
