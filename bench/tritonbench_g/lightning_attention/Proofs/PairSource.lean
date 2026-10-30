import bench.tritonbench_g.lightning_attention.IO
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.PairSource
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def kernel (stage : Stage) (p : Parameters) : ComputeKernel :=
  match stage with
  | .forward => forwardCandidate p
  | .intra => intraCandidate p
  | .inter => interCandidate p

def original (stage : Stage) (p : Parameters) : ComputeKernel :=
  match stage with
  | .forward => forward p
  | .intra => intra p
  | .inter => inter p

def prefixLength : Stage → Nat
  | .forward | .intra => 6
  | .inter => 5

def prelude (stage : Stage) (p : Parameters) := (original stage p).surfaceBody.take (prefixLength stage)
def candidatePrefix (stage : Stage) (p : Parameters) := (kernel stage p).surfaceBody.take (prefixLength stage)
def suffix (stage : Stage) (p : Parameters) := (original stage p).surfaceBody.drop (prefixLength stage)

theorem split (stage : Stage) (p : Parameters) :
    (kernel stage p).surfaceBody = candidatePrefix stage p ++ suffix stage p ∧
    (original stage p).surfaceBody = prelude stage p ++ suffix stage p := by
  cases stage <;> constructor <;> rfl

/-- CSE of the exact integer base offset, before any memory access. No floating
operation, conversion, reduction, branch or store is changed. -/
theorem prefix_eq (M : Algebra α) (stage : Stage) (p : Parameters) (s : State α) :
    run M (candidatePrefix stage p) s = run M (prelude stage p) s := by
  cases stage <;>
    simp [candidatePrefix,prelude,prefixLength,kernel,original,
      forwardCandidate,intraCandidate,interCandidate,forward,intra,inter,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,bop,Region.cast,ComputeDType.eraseDType,Option.bind]

theorem exact (M : Algebra α) (stage : Stage) (p : Parameters) (s : State α) :
    FP.Structural.exec M (kernel stage p) s = FP.Structural.exec M (original stage p) s := by
  change run M (kernel stage p).surfaceBody s = run M (original stage p).surfaceBody s
  rw [(split stage p).1,(split stage p).2,run_append,run_append,prefix_eq]

theorem prefix_safe (M : Algebra α) (stage : Stage) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (prelude stage p) s ∧
      Safety.traceSafeList M bounds (candidatePrefix stage p) s := by
  cases stage <;>
    simp [candidatePrefix,prelude,prefixLength,kernel,original,
      forwardCandidate,intraCandidate,interCandidate,forward,intra,inter,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,bop,Region.cast,ComputeDType.eraseDType,Option.bind]

theorem safe_iff (M : Algebra α) (stage : Stage) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) :
    TraceSafe M bounds (kernel stage p) s ↔ TraceSafe M bounds (original stage p) s := by
  change Safety.traceSafeList M bounds (kernel stage p).surfaceBody s ↔
    Safety.traceSafeList M bounds (original stage p).surfaceBody s
  rw [(split stage p).1,(split stage p).2]
  simp only [Safety.traceSafeList_append,prefix_eq,
    (prefix_safe M stage p s bounds).1,(prefix_safe M stage p s bounds).2,true_and]

#axiomsClean exact
#axiomsClean safe_iff
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.PairSource
