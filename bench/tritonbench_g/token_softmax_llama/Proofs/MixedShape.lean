import bench.tritonbench_g.token_softmax_bloom.Proofs.MixedShape
import bench.tritonbench_g.token_softmax_llama.TokenSoftmaxLlama
namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama.MixedTyped
open VeriTile.Triton TokenSoftmaxBloom.Typed TokenSoftmaxBloom.MixedTyped
/-- Llama has the same complete JIT kernel with an independent destination tensor. -/
theorem source_shape (d out : Storage) (p : Parameters) :
    (kernel d out p).eraseDType.surfaceBody =
      (token_softmax_surface p.input p.start p.length p.output p.inputHeadStride
        p.inputTokenStride p.outputHeadStride p.outputTokenStride p.width).eraseDType.surfaceBody := by
  exact TokenSoftmaxBloom.MixedTyped.source_shape d out p
#axiomsClean source_shape
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama.MixedTyped
