import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedCumExecution
import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeRealSafety

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram RebasedCumAccess RebasedCumExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

structure Bounds (p : Layout) (l : Launch) (c : Nat) (s o : RegionName) (bounds : RegionBounds) : Prop where
  input : RealHAccessSafety.MatrixBounds s (tileBase p l c) p.BT p.BS p.BT p.BS p.timeStride p.channelStride 0 0 bounds
  output : RealHAccessSafety.MatrixBounds o (tileBase p l c) p.BT p.BS p.BT p.BS p.timeStride p.channelStride 0 0 bounds

theorem source_safe (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool)
    (dtype out_dtype : FloatDType) (bounds : RegionBounds) (u : BlockState)
    (hb : enabled p heads base u → Bounds p (decoded p base u) (chunk p base u) s o bounds) :
    (RebasedCumProgram.source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel.TraceSafe bounds u := by
  apply CumulativeSource.Real.source_safe (RebasedCumProgram.parameters p heads base swapped)
    dtype out_dtype s o bounds u
  intro ha
  have ha' : enabled p heads base u := ha
  refine ⟨?_,?_⟩
  · simpa [RebasedCumProgram.parameters,CumulativeSource.Real.tileBase,CumulativeSource.Real.headIndex,
      CumulativeSource.Real.chunkIndex,CumulativeSource.Real.featureIndex,CumulativeSource.Real.linearIndex,
      tileBase,decoded,chunk,linear] using (hb ha').input
  · simpa [RebasedCumProgram.parameters,CumulativeSource.Real.tileBase,CumulativeSource.Real.headIndex,
      CumulativeSource.Real.chunkIndex,CumulativeSource.Real.featureIndex,CumulativeSource.Real.linearIndex,
      tileBase,decoded,chunk,linear] using (hb ha').output

theorem source_flatten_ok (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool)
    (dtype out_dtype : FloatDType) :
    (RebasedCumProgram.source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel.FlattenOk :=
  CumulativeSource.Real.source_flatten_ok (RebasedCumProgram.parameters p heads base swapped)
    dtype out_dtype s o

#axiomsClean source_safe
#axiomsClean source_flatten_ok
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumSafety
