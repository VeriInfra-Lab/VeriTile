import bench.tritonbench_g.matrix_vector_multip.Proofs.TypedShape
import bench.tritonbench_g.matrix_vector_multip.Proofs.TypedSoftware
namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.FPCorrect
open VeriTile.Triton FP.Structural Typed
noncomputable section
/-- Full original looping kernel in the one-block profile, including zero
columns. Every cast, initial +0 add and reduction remains in the value graph. -/
specification mv_one_block_typed_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (d : FloatDType) (p : Parameters) (s : State α)
    (hw : 0 < p.blockColumns) (hcols : p.columns ≤ p.blockColumns)
    (hinj : Function.Injective (fun i : Fin p.blockRows => rowIndex p s i * p.strideOut)) :
    ∃ t, FP.Structural.exec M (kernel d p) s = some t ∧
      t.mem = (reference M d p s).mem ∧
      (∀ i : TileIndex [p.blockRows,1], rowIndex p s i.1 < p.rows →
        t.mem p.out (outputOffset p s i) = .mk d.toTileDType (ofFloat d (outputValues M d p s i))) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  refine ⟨(execution M d p s hw hcols).val, (execution M d p s hw hcols).property,
    execution_memory M d p s hw hcols, ?_, ?_⟩
  · intro i hi
    rw [execution_memory]
    exact reference_readback M d p s hinj i hi
  · intro r o h
    rw [execution_memory]
    exact reference_frame M d p s r o h

/-- Concrete binary32 scalar arithmetic and explicitly sequential software
reduction. The native kernel tree must be bound separately. -/
specification mv_one_block_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (hw : 0 < p.blockColumns) (hcols : p.columns ≤ p.blockColumns)
    (hinj : Function.Injective (fun i : Fin p.blockRows => rowIndex p s i * p.strideOut)) :
    ∃ t, FP.Structural.exec (backend rest cfg) (kernel .fp32 p) s = some t ∧
      (∀ i : TileIndex [p.blockRows,1], rowIndex p s i.1 < p.rows →
        t.mem p.out (outputOffset p s i) = .mk .fp32 (roundedOutput cfg p s i)) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, _, hv, hf⟩ := mv_one_block_typed_correctness (backend rest cfg) .fp32 p s hw hcols hinj
  refine ⟨t, he, ?_, hf⟩
  intro i hi
  simpa [output_software, ofFloat] using hv i hi
#axiomsClean mv_one_block_typed_correctness
#axiomsClean mv_one_block_software_fp32
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.FPCorrect
