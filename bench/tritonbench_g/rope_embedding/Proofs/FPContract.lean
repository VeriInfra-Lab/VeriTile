import bench.tritonbench_g.rope_embedding.Proofs.FPHalves
namespace VeriTile.Bench.TritonBenchG.RopeEmbedding.FPHalves
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

theorem offset_injective {α : Type} (p : Parameters) (s : State α) (second : Bool) :
    Function.Injective (offset p s second) := by
  intro i j h
  have ha := Nat.add_left_cancel (Nat.add_right_cancel h)
  exact Prod.ext (Fin.ext ha) (Subsingleton.elim _ _)

def Post {α : Type} (second : Bool) (p : Parameters) (initial final : State α)
    (results : TileIndex [p.width] → α) : Prop :=
  (∀ i, active p initial i → final.mem p.input (offset p initial second i) = .mk .fp32 (results i)) ∧
  (∀ r o, (r ≠ p.input ∨ ∀ i : TileIndex [p.width], active p initial i → o ≠ offset p initial second i) →
    final.mem r o = initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (backward second : Bool)
    (p : Parameters) (s : State α) :
    ∃ final, FP.Structural.exec M (source backward second p) s = some final ∧
      Post second p s final (value M backward second p s) := by
  refine ⟨(execution M backward second p s).val, (execution M backward second p s).property, ?_, ?_⟩
  · intro i hi
    rw [execution_memory]
    exact State.masked_scatter_readback _ _ _ _ _ i hi (offset_injective p s second)
  · intro r o h
    rw [execution_memory]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (h.imp id (fun h i _ => h i)) _
#axiomsClean offset_injective
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.RopeEmbedding.FPHalves
