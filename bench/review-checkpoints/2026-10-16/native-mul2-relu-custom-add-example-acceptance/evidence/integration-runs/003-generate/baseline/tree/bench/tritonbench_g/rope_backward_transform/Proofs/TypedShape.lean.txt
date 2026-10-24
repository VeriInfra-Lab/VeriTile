import bench.tritonbench_g.rope_transform.Proofs.TypedShape
import bench.tritonbench_g.rope_backward_transform.RopeBackwardTransform
namespace VeriTile.Bench.TritonBenchG.RopeBackwardTransform.Typed
open VeriTile.Triton RopeTransform.Typed
/-- Both benchmark Python entries use the same two-branch JIT body. -/
theorem legacy_source_same (p : Parameters) :
    triton_rope_surface p.q p.k p.cos p.sin p.qStride p.kStride p.cosStride
      p.sinStride p.length 0 p.qHeads p.kHeads p.headDim p.paddedQHeads
      p.paddedKHeads p.paddedHeadDim 0 p.backward =
    RopeTransform.triton_rope_surface p.q p.k p.cos p.sin p.qStride p.kStride p.cosStride
      p.sinStride p.length 0 p.qHeads p.kHeads p.headDim p.paddedQHeads
      p.paddedKHeads p.paddedHeadDim 0 p.backward := by rfl

theorem source_shape (p : Parameters) :
    (kernel .fp32 .fp32 p).eraseDType.surfaceBody =
      (triton_rope_surface p.q p.k p.cos p.sin p.qStride p.kStride p.cosStride
        p.sinStride p.length 0 p.qHeads p.kHeads p.headDim p.paddedQHeads
        p.paddedKHeads p.paddedHeadDim 0 p.backward).eraseDType.surfaceBody := by
  rw [legacy_source_same]
  exact RopeTransform.Typed.source_shape p
#axiomsClean legacy_source_same
#axiomsClean source_shape
end VeriTile.Bench.TritonBenchG.RopeBackwardTransform.Typed
