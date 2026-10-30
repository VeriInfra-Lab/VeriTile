import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Grid

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.RealExecution
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section

/-- Complete raw kernel execution and independent rotary/cache formula. Config
contains shape and decoded length/table arrays; its validity gives only kernel
index bounds and write separation. -/
theorem fused_rotary_real (rest : Algebra ℝ) (c : Config) (valid : c.Valid)
    (initial : Memory ℝ) :
    ∃ final,Source.grid rest realOps (Source.parameters c.shape) (programs c.shape)
        (Source.lift c initial)=some final ∧
      (∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width →
        (final (Source.region kind.output) (logicalOutput c kind t h d)).read .real=
          formula c.shape initial kind t h d) ∧
      (∀ r a,¬Footprint c (Source.decoded r) a → final r a=Source.lift c initial r a) ∧
      (c.shape.tokens=0 → final=Source.lift c initial) := by
  obtain ⟨final,executed,values,frame⟩ := Source.complete_real rest c valid initial
  refine ⟨final,executed,values,frame,?_⟩
  intro empty
  have noPrograms : programs c.shape=[] := by simp [programs,empty]
  rw [noPrograms,Source.grid] at executed
  exact Option.some.inj executed.symm

#axiomsClean fused_rotary_real
#specNonCircular formula avoiding [Source.grid,Source.kernel,actualSource]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.RealExecution
