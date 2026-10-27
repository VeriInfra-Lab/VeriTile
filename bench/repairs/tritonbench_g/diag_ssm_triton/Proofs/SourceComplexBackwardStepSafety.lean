import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardCompose
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardIndexedSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardPreviousSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardSuffixSafety
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section

theorem backward_step_safe (p : BackwardLayout) (length t : Nat) (a b : BlockState)
    (ht : t < length) (h : BackwardRelated p length a b) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (forwardLoop p length)
      (a.setReg "i" .int [] (Tile.scalar (t : Int))) ↔
    Stmt.TraceSafeList bounds (historicalLoop p length)
      (b.setReg "i" .nat [] (Tile.scalar t)) := by
  obtain ⟨a₁, b₁, ha₁, hb₁, h₁⟩ := indexed_steps p length t a b h ht
  obtain ⟨a₂, b₂, ha₂, hb₂, h₂⟩ := previous_steps p length t a₁ b₁ h₁
  conv_lhs => rw [split_loop (forwardLoop p length)]
  conv_rhs => rw [split_loop (historicalLoop p length)]
  simp only [List.append_assoc, TraceRewrite.append_iff, ha₁, hb₁, Option.some.injEq, forall_eq',
    ha₂, hb₂, indexed_safe p length t a b h ht bounds, previous_safe p length t a₁ b₁ h₁ bounds,
    suffix_safe p length t a₂ b₂ h₂ bounds]

#axiomsClean backward_step_safe
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
