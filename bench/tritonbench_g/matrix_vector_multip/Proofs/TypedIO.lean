import bench.tritonbench_g.matrix_vector_multip.Proofs.TypedContract
namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
open VeriTile.Triton FP.Structural
noncomputable section

def Outside {α : Type} (p : Parameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.out ∨ ∀ i : TileIndex [p.blockRows,1], rowIndex p s i.1 < p.rows → o ≠ outputOffset p s i

theorem offset_injective {α : Type} (p : Parameters) (s : State α)
    (hinj : Function.Injective (fun i : Fin p.blockRows => rowIndex p s i * p.strideOut)) :
    Function.Injective (outputOffset p s) := by
  intro i j h
  exact Prod.ext (hinj h) (Subsingleton.elim _ _)

theorem reference_readback {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α)
    (hinj : Function.Injective (fun i : Fin p.blockRows => rowIndex p s i * p.strideOut))
    (i : TileIndex [p.blockRows,1]) (hi : rowIndex p s i.1 < p.rows) :
    (reference M d p s).mem p.out (outputOffset p s i) =
      .mk d.toTileDType (ofFloat d (outputValues M d p s i)) := by
  exact State.masked_scatter_readback _ _ _ _ _ i hi (offset_injective p s hinj)

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat) (h : Outside p s r o) :
    (reference M d p s).mem r o = s.mem r o := by
  apply State.masked_scatter_frame
  rcases h with h | h
  · exact Or.inl h
  · exact Or.inr (fun i _ hi => h i hi)
#axiomsClean offset_injective
#axiomsClean reference_readback
#axiomsClean reference_frame
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
