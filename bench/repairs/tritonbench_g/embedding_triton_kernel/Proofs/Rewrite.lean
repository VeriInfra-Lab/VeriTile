import bench.repairs.tritonbench_g.embedding_triton_kernel.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.Rewrite
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (p : Parameters) : Program where
  kernel := source .int32 p
  domain := []

def indexCommuted (p : Parameters) : Program where
  kernel := indexCommutedSource .int32 p
  domain := []

/-- The actual integer-product variant preserves the entire typed execution,
including all iterations, memory, registers and any failure. -/
theorem equivalent (p : Parameters) : original p ≡[assumptions] indexCommuted p := by
  equiv_decompose <;> fp_prove

/-- Direct whole-state equality for every algebra, so constructive success
can be transported without introducing a numerical-model premise. -/
theorem execution_eq {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Parameters) (s : FP.Structural.State α) :
    FP.Structural.exec M (source .int32 p) s =
      FP.Structural.exec M (indexCommutedSource .int32 p) s := by
  have firstStep : FP.Structural.run M ((source .int32 p).surfaceBody.take 1) s =
      FP.Structural.run M ((indexCommutedSource .int32 p).surfaceBody.take 1) s := by
    simp [source,indexCommutedSource,ComputeKernel.surfaceBody,FP.Structural.run,
      FP.Structural.step,FP.Structural.evalExpr,
      FP.Structural.evalOp_unfold,FP.Structural.numeric,
      ComputeDType.eraseDType,Option.bind]
    congr 1
    funext i
    exact Int.mul_comm _ _
  change FP.Structural.run M (source .int32 p).surfaceBody s =
    FP.Structural.run M (indexCommutedSource .int32 p).surfaceBody s
  rw [←List.take_append_drop 1 (source .int32 p).surfaceBody,
    ←List.take_append_drop 1 (indexCommutedSource .int32 p).surfaceBody]
  rw [FP.Structural.run_append,FP.Structural.run_append,firstStep]
  rfl

#print_fp_assumptions equivalent
#axiomsClean equivalent
#axiomsClean execution_eq
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.Rewrite
