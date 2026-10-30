import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RawPair
open VeriTile.Triton
open Source
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem prefix_run (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block)
    {α : Type} [Inhabited α] (A : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    FP.Structural.run A ((kernel f r stride M N eps p block hb).surfaceBody.take 2) s =
      FP.Structural.run A ((indexCommuted f r stride M N eps p block hb).surfaceBody.take 2) s := by
  by_cases hN : N = 1
  all_goals simp [kernel,indexCommuted,hN,rawKernel,rawIndexCommuted,rawKernelOne,rawIndexCommutedOne,
    ComputeKernel.surfaceBody,FP.Structural.run,FP.Structural.step,
    FP.Structural.evalExpr,FP.Structural.evalOp_unfold,FP.Structural.numeric]
  all_goals congr 1
  all_goals funext i
  all_goals simp [FP.Structural.bop,Int.mul_comm]

theorem suffix (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) :
    (kernel f r stride M N eps p block hb).surfaceBody.drop 2 =
      (indexCommuted f r stride M N eps p block hb).surfaceBody.drop 2 := by
  by_cases hN : N = 1 <;> simp only [kernel,indexCommuted,hN,ite_true,ite_false] <;> rfl

/-- Complete actual source equivalence for arbitrary primitive interpretation,
including every state component and failure. Successful IO is proved separately. -/
theorem whole_state (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block)
    {α : Type} [Inhabited α] (A : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    FP.Structural.exec A (kernel f r stride M N eps p block hb) s =
      FP.Structural.exec A (indexCommuted f r stride M N eps p block hb) s := by
  change FP.Structural.run A (kernel f r stride M N eps p block hb).surfaceBody s =
    FP.Structural.run A (indexCommuted f r stride M N eps p block hb).surfaceBody s
  conv_lhs => rw [← List.take_append_drop 2 (kernel f r stride M N eps p block hb).surfaceBody]
  conv_rhs => rw [← List.take_append_drop 2 (indexCommuted f r stride M N eps p block hb).surfaceBody]
  simp only [FP.Structural.run_append,prefix_run,suffix]

#axiomsClean prefix_run
#axiomsClean whole_state
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RawPair
