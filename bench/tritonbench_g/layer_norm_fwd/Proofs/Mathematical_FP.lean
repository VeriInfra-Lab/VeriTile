import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Source
import VeriTile.Triton.Float.Structural

/-! Full-state FP equivalence checkpoint for the complete fp32 source pair.
The theorem includes all flags, all outputs, both seven-round RNG streams,
eager arithmetic and every failure. Other storage profiles and successful
execution/correctness remain separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 4000

namespace FPProof

theorem prefix_run (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    {α : Type} [Inhabited α] (A : FP.Structural.Algebra α) (u : FP.Structural.State α) :
    FP.Structural.run A ((original f r stride M N eps p block hBlock).surfaceBody.take 2) u =
      FP.Structural.run A ((candidate f r stride M N eps p block hBlock).surfaceBody.take 2) u := by
  simp [original, candidate, ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalOp_unfold, FP.Structural.numeric]
  congr 1
  funext i
  simp [FP.Structural.bop, Nat.mul_comm]

theorem suffix (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    (original f r stride M N eps p block hBlock).surfaceBody.drop 2 =
      (candidate f r stride M N eps p block hBlock).surfaceBody.drop 2 := rfl

/-- Equality of the complete typed executions for every structural FP algebra,
including failure and every state field. No numerical or operand-domain law
is assumed. This theorem does not assert existence of a successful run. -/
theorem whole_state (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    {α : Type} [Inhabited α] (A : FP.Structural.Algebra α) (u : FP.Structural.State α) :
    FP.Structural.exec A (original f r stride M N eps p block hBlock) u =
      FP.Structural.exec A (candidate f r stride M N eps p block hBlock) u := by
  change FP.Structural.run A (original f r stride M N eps p block hBlock).surfaceBody u =
    FP.Structural.run A (candidate f r stride M N eps p block hBlock).surfaceBody u
  conv_lhs => rw [← List.take_append_drop 2 (original f r stride M N eps p block hBlock).surfaceBody]
  conv_rhs => rw [← List.take_append_drop 2 (candidate f r stride M N eps p block hBlock).surfaceBody]
  simp only [FP.Structural.run_append, prefix_run, suffix]

end FPProof
end
end VeriTile.Bench.TritonBenchG.LayerNormFwd
