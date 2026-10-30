import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardSuffix
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceLoopSimulation
import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardProjection
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

/-- A proof-only partition of a loop; the authoritative source is untouched. -/
theorem split_loop (xs : List Stmt) :
    xs = xs.take 3 ++ (xs.drop 3).take 1 ++ xs.drop 4 := by
  have hd : (xs.drop 3).drop 1 = xs.drop 4 := List.drop_drop
  rw [← hd]
  rw [List.append_assoc, List.take_append_drop, List.take_append_drop]

theorem backward_step_related (p : BackwardLayout) (length t : Nat) (a b : BlockState)
    (ht : t < length) (h : BackwardRelated p length a b) :
    ∃ aa bb, stepStmts (forwardLoop p length)
        (a.setReg "i" .int [] (Tile.scalar (t : Int))) = some aa ∧
      stepStmts (historicalLoop p length)
        (b.setReg "i" .nat [] (Tile.scalar t)) = some bb ∧ BackwardRelated p length aa bb := by
  obtain ⟨a₁, b₁, ha₁, hb₁, h₁⟩ := indexed_steps p length t a b h ht
  obtain ⟨a₂, b₂, ha₂, hb₂, h₂⟩ := previous_steps p length t a₁ b₁ h₁
  obtain ⟨a₃, b₃, ha₃, hb₃, h₃⟩ := suffix_steps p length t a₂ b₂ h₂
  refine ⟨a₃, b₃, ?_, ?_, h₃⟩
  · conv_lhs => rw [split_loop (forwardLoop p length)]
    simp only [ProgramRewrite.append, ha₁, Option.bind_some, ha₂, ha₃]
  · conv_lhs => rw [split_loop (historicalLoop p length)]
    simp only [ProgramRewrite.append, hb₁, Option.bind_some, hb₂, hb₃]

#axiomsClean split_loop
#axiomsClean backward_step_related
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
