import bench.tritonbench_g.token_softmax_bloom.Proofs.TypedShape
import bench.tritonbench_g.token_softmax_llama.TokenSoftmaxLlama
namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama.Typed
open VeriTile.Triton TokenSoftmaxBloom.Typed
/-- Llama uses the same complete kernel; its host storage profile is fp32. -/
theorem source_shape (d : FloatDType) (p : Parameters) :
    (kernel d p).eraseDType.surfaceBody =
      (token_softmax_surface p.input p.start p.length p.output p.inputHeadStride
        p.inputTokenStride p.outputHeadStride p.outputTokenStride p.width).eraseDType.surfaceBody := by
  exact TokenSoftmaxBloom.Typed.source_shape d p
#axiomsClean source_shape
end VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama.Typed
