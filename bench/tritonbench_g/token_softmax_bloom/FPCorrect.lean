import bench.tritonbench_g.token_softmax_bloom.Proofs.TypedIO
import bench.tritonbench_g.token_softmax_bloom.Proofs.TypedShape
namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.FPCorrect
open VeriTile.Triton FP.Structural Typed
noncomputable section
/-- Full row computation and masked writeback, with independent initial-state
value formula. Every primitive retains its precision and storage conversion.
The native max/exp/sum implementation is a backend binding obligation. -/
specification token_softmax_typed_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (d : FloatDType) (p : Parameters) (s : State α)
    (hwidth : 0 < p.width) (hstride : p.outputTokenStride ≠ 0) :
    ∃ t, FP.Structural.exec M (kernel d p) s = some t ∧
      t.mem = (reference M d p s).mem ∧
      (∀ i : TileIndex [p.width], i.1.val < count p s →
        t.mem p.output (outputOffset p s i) =
          .mk d.toTileDType (ofFloat d (outputValues M d p s i))) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  refine ⟨(execution M d p s hwidth).val, (execution M d p s hwidth).property,
    execution_memory M d p s hwidth, ?_, ?_⟩
  · intro i hi
    rw [execution_memory]
    exact reference_readback M d p s hstride i hi
  · intro r o h
    rw [execution_memory]
    exact reference_frame M d p s r o h
#axiomsClean token_softmax_typed_correctness
end
end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.FPCorrect
