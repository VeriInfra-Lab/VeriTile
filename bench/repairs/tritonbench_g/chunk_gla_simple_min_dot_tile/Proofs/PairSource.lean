import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.Typed
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PairSource
open VeriTile Triton FP.Structural Host
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def prelude (c : Shape) (scale : ℝ) :=
  (kernel c .fp32 .fp32 .fp32 .fp32 scale).surfaceBody.take 8
def candidatePrefix (c : Shape) (scale : ℝ) :=
  (rowReusedKernel c scale).surfaceBody.take 8
def suffix (c : Shape) (scale : ℝ) :=
  (kernel c .fp32 .fp32 .fp32 .fp32 scale).surfaceBody.drop 8

theorem source_split (c : Shape) (scale : ℝ) :
    (rowReusedKernel c scale).surfaceBody = candidatePrefix c scale ++ suffix c scale ∧
    (kernel c .fp32 .fp32 .fp32 .fp32 scale).surfaceBody = prelude c scale ++ suffix c scale := by
  constructor <;> rfl

/-- The same generated lane vector supplies the mask and row addresses.
No floating operation or payload value is rewritten. -/
theorem prefix_eq (M : Algebra α) (c : Shape) (scale : ℝ) (s : State α) :
    run M (candidatePrefix c scale) s = run M (prelude c scale) s := by
  simp [candidatePrefix,prelude,rowReusedKernel,kernel,source,fp32Original,rowReusedSource,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,numericLe,natLe,bop,Region.cast,ComputeDType.eraseDType,Option.bind]

theorem execution_eq (M : Algebra α) (c : Shape) (scale : ℝ) (s : State α) :
    exec M (rowReusedKernel c scale) s =
      exec M (kernel c .fp32 .fp32 .fp32 .fp32 scale) s := by
  change run M (rowReusedKernel c scale).surfaceBody s =
    run M (kernel c .fp32 .fp32 .fp32 .fp32 scale).surfaceBody s
  rw [(source_split c scale).1,(source_split c scale).2,run_append,run_append,prefix_eq]

theorem prefix_safe (M : Algebra α) (c : Shape) (scale : ℝ) (s : State α)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (prelude c scale) s ∧
      Safety.traceSafeList M bounds (candidatePrefix c scale) s := by
  simp [candidatePrefix,prelude,rowReusedKernel,kernel,source,fp32Original,rowReusedSource,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,numericLe,natLe,bop,Region.cast,ComputeDType.eraseDType,Option.bind]

theorem safe_iff (M : Algebra α) (c : Shape) (scale : ℝ) (s : State α)
    (bounds : RegionName → Nat) :
    TraceSafe M bounds (rowReusedKernel c scale) s ↔
      TraceSafe M bounds (kernel c .fp32 .fp32 .fp32 .fp32 scale) s := by
  change Safety.traceSafeList M bounds (rowReusedKernel c scale).surfaceBody s ↔
    Safety.traceSafeList M bounds (kernel c .fp32 .fp32 .fp32 .fp32 scale).surfaceBody s
  rw [(source_split c scale).1,(source_split c scale).2]
  simp only [Safety.traceSafeList_append,prefix_eq,
    (prefix_safe M c scale s bounds).1,(prefix_safe M c scale s bounds).2,true_and]

#axiomsClean execution_eq
#axiomsClean safe_iff
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PairSource
