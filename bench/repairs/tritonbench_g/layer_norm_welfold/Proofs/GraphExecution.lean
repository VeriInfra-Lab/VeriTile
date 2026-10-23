import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelGraph
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelBoundary

namespace VeriTile.Bench.Repairs.LayerNormWelfold.KernelGraph
open VeriTile.Triton FP.Structural Reuse Native
noncomputable section

/-- Successful typed-source computation of the independent operation graph.
The native profile subsequently observes this graph with actual FMA semantics. -/
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (p : Primitives α) (bindings : SourceBindings M p) (s : State α) (x w bias : Nat → α)
    (S D : Nat) (Y Mean Rstd : RegionName) (c : Configuration)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd)
    (hx : ∀ row, row < S → ∀ col, col < D → (s.mem "X" (col + D * row)).read .real = x (col + D * row))
    (hw : ∀ col, col < D → (s.mem "W" col).read .real = w col)
    (hb : ∀ col, col < D → (s.mem "Bias" col).read .real = bias col) :
    ∃ final, Source.launch (withEquality M eq32) s Y Mean Rstd S D c.block (SourceBinding.tree c.tree) = some final ∧
      (∀ row, row < S → final.mem Mean row = .mk .real
        (eval p (logicalInputs M x w bias D row) (meanExpr D c.block c.tree))) ∧
      (∀ row, row < S → final.mem Rstd row = .mk .real
        (eval p (logicalInputs M x w bias D row) (inverseExpr D c.block c.tree))) ∧
      (∀ row, row < S → ∀ col, col < D → final.mem Y (col + D * row) = .mk .bf16
        (eval p (logicalInputs M x w bias D row) (outputExpr D c.block col c.tree))) ∧
      (∀ q off, gridMiss Y Mean Rstd S D q off → final.mem q off = s.mem q off) := by
  have agree := fun row hr => observed_agreement M s x w bias D row (hx row hr) hw hb
  obtain ⟨final, execution, effect⟩ := KernelBoundary.complete M eq32 s Y Mean Rstd S D c sep
  refine ⟨final, execution, ?_, ?_, ?_, ?_⟩
  · intro row hr
    rw [effect.mean row (List.mem_range.mpr hr), ←mean_eval M p bindings s "X" "W" "Bias" D c.block row c.tree]
    congr 1
    exact mean_congr p _ _ D c.block (agree row hr) c.tree
  · intro row hr
    rw [effect.rstd row (List.mem_range.mpr hr), ←inverse_eval M p bindings s "X" "W" "Bias" D c.block row c.tree]
    congr 1
    exact inverse_congr p _ _ D c.block (agree row hr) c.tree
  · intro row hr col hc
    rw [effect.output row (List.mem_range.mpr hr) col hc, ←output_eval M p bindings s "X" "W" "Bias" D c.block row col c.tree]
    congr 1
    exact output_congr p _ _ D c.block col (agree row hr) c.tree hc
  · intro q off miss
    exact effect.frame q off (fun row hr => miss row (List.mem_range.mp hr))

#axiomsClean complete
end
end VeriTile.Bench.Repairs.LayerNormWelfold.KernelGraph
