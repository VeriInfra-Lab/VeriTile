import bench.tritonbench_g.token_softmax_bloom.Proofs.MixedIO
import bench.tritonbench_g.token_softmax_bloom.Proofs.MixedShape
namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.MixedFPCorrect
open VeriTile.Triton FP.Structural TokenSoftmaxBloom.Typed TokenSoftmaxBloom.MixedTyped
noncomputable section
/-- The whole selected row succeeds for independent fp16/bf16/fp32 input/output
storage, including the original fp32 arithmetic graph and masked destination cast.
The initial-state value formula and off-write frame are independent of execution.
Native scalar conformance, launch composition and other dtype routes stay open. -/
specification token_softmax_mixed_output_typed_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (d out : Storage) (p : Parameters) (s : State α)
    (hwidth : 0 < p.width) (hstride : p.outputTokenStride ≠ 0) :
    (TokenSoftmaxBloom.MixedTyped.kernel d out p).eraseDType.surfaceBody =
      (token_softmax_surface p.input p.start p.length p.output p.inputHeadStride
        p.inputTokenStride p.outputHeadStride p.outputTokenStride p.width).eraseDType.surfaceBody ∧
    ∃ t, FP.Structural.exec M (TokenSoftmaxBloom.MixedTyped.kernel d out p) s = some t ∧
      t.mem = (TokenSoftmaxBloom.MixedTyped.reference M d out p s).mem ∧
      (∀ i : TileIndex [p.width], i.1.val < count p s →
        t.mem p.output (outputOffset p s i) =
          .mk (dtype out).toTileDType (ofFloat (dtype out) (TokenSoftmaxBloom.MixedTyped.outputValues M d out p s i))) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  refine ⟨TokenSoftmaxBloom.MixedTyped.source_shape d out p, ?_⟩
  refine ⟨(TokenSoftmaxBloom.MixedTyped.execution M d out p s hwidth).val,
    (TokenSoftmaxBloom.MixedTyped.execution M d out p s hwidth).property,
    TokenSoftmaxBloom.MixedTyped.execution_memory M d out p s hwidth, ?_, ?_⟩
  · intro i hi
    rw [TokenSoftmaxBloom.MixedTyped.execution_memory]
    exact TokenSoftmaxBloom.MixedTyped.reference_readback M d out p s hstride i hi
  · intro r o h
    rw [TokenSoftmaxBloom.MixedTyped.execution_memory]
    exact TokenSoftmaxBloom.MixedTyped.reference_frame M d out p s r o h
#axiomsClean token_softmax_mixed_output_typed_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.MixedFPCorrect
