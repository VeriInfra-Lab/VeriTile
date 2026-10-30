import bench.tritonbench_g.triton_attention.Proofs.MemoryAgreement

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.MemoryAgreement
open VeriTile Triton FP.Structural Source
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000

theorem preprocess_step (P : Primitives α) (c : Shape) (pid : Grid.Program)
    (legal : Grid.Legal (Grid.preprocessGrid c) pid) (a b : Memory α) (h : Agree c.elements a b) :
    Agree c.elements (Grid.preprocessStep P c pid a) (Grid.preprocessStep P c pid b) := by
  have input (r : RegionName) : preInput P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid a) r=
      preInput P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid b) r := by
    funext i
    apply congrArg (fun cell : Cell α=>P.cast (some .fp32) .fp32 .real (cell.read .fp32))
    exact h r _ (SourceAccess.preprocess_element_bound c (Grid.entry (Grid.preprocessGrid c) pid a) legal.1 i)
  have denom : preDenominator P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid a)=
      preDenominator P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid b) := by
    funext i
    apply congrArg (fun cell : Cell α=>P.cast (some .fp32) .fp32 .real (cell.read .fp32))
    exact h "L" _ (lt_of_lt_of_le
      (SourceAccess.preprocess_row_bound c (Grid.entry (Grid.preprocessGrid c) pid a) legal.1 i) (row_capacity c))
  have quotient : preQuotient P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid a)=
      preQuotient P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid b) := by
    funext i
    simp only [preQuotient,input,denom]
  have gradient : preGradientCell P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid a)=
      preGradientCell P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid b) := by
    funext i
    simp only [preGradientCell,quotient]
  have products : preProducts P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid a)=
      preProducts P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid b) := by
    funext i
    simp only [preProducts,input,quotient]
  have delta : preDeltaCell P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid a)=
      preDeltaCell P (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid b) := by
    funext i
    simp only [preDeltaCell,preDelta,products]
  unfold Grid.preprocessStep preWrite
  dsimp only []
  rw [gradient,delta]
  apply scatterState
  exact scatterState _ _ _ h _ _ _ _

#axiomsClean preprocess_step
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.MemoryAgreement
