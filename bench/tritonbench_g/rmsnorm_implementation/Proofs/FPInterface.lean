import bench.tritonbench_g.rmsnorm_implementation.PairIO
import bench.tritonbench_g.rmsnorm_implementation.Proofs.FPSuccess
import bench.tritonbench_g.cache_transform_triton.Sources.Mathematical_IndexRewrite
import VeriTile.Triton.Float.ProfiledMemory

/-! Kernel-only equivalence of the two canonical typed sources. Both complete
executions succeed and agree on all typed memory, with no floating rewrite
assumption. The observed native FMA/reduction graph is exported separately and
its compiler correspondence remains explicit in Proofs/KernelValues. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.FPInterface
open VeriTile Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Exact natural-number address multiplication, preserving all FP operations,
casts, reduction order, masks, arbitrary padding values, and both loops. -/
theorem source_runs_equal {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (sxb sxm sxk sw sob som sok n B : Nat) (eps : ℝ) (s : FP.Structural.State α) :
    FP.Structural.exec (FP.ProfiledRewrite.engine M)
      (kernel "x" "weights" "out" sxb sxm sxk sw sob som sok n B eps) s =
    FP.Structural.exec (FP.ProfiledRewrite.engine M)
      (indexCommutedKernel "x" "weights" "out" sxb sxm sxk sw sob som sok n B eps) s := by
  simp [kernel,indexCommutedKernel,ComputeKernel.surfaceBody,FP.Structural.exec,FP.Structural.run,FP.Structural.step,
    FP.Structural.evalExpr,FP.Structural.evalOp_unfold,FP.Structural.numeric,
    FP.Structural.bop,FP.Structural.State.setReg,Option.bind,Nat.mul_comm]
  unfold FP.Structural.bop
  simp only [Nat.mul_comm]

theorem rmsnorm_index_commuted_equiv (sxb sxm sxk sw sob som sok n B : Nat) (eps : ℝ) :
    original sxb sxm sxk sw sob som sok n B eps ≡[assumptions]
      commuted sxb sxm sxk sw sob som sok n B eps := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM s hd
  obtain ⟨final,ran⟩ := FPSuccess.success (FP.ProfiledRewrite.engine M) sxb sxm sxk sw sob som sok n B eps s
  exact ⟨final,final,ran,(source_runs_equal M sxb sxm sxk sw sob som sok n B eps s).symm.trans ran,rfl⟩

#print_fp_assumptions rmsnorm_index_commuted_equiv
#axiomsClean rmsnorm_index_commuted_equiv
#axiomsClean source_runs_equal
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.FPInterface
