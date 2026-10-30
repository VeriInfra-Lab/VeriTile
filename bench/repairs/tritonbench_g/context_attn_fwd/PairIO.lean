import bench.repairs.tritonbench_g.context_attn_fwd.NativeIO
import VeriTile.Triton.Float.ArrayProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.ContextAttnFwd.PairIO
open VeriTile.Triton FP.Structural NativeFormula
noncomputable section
variable {α : Type} [Inhabited α]

def candidateInvocation (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (job : Launch) : Invocation α :=
  { NativeArrayBridge.invocation p r scale M job with kernel := blockEndFactoredFP32 p r scale }

def candidateVariant (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) :
    ArrayProgramIO.Variant (NativeIO.io p r M dots scale jobs) where
  schedule := fun _ => jobs.map (candidateInvocation p r scale M)
  observe := fun _ _ => id

def original (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) :=
  ArrayProgramIO.Implementation.original (NativeIO.io p r M dots scale jobs) (fun _ => True)
def blockEndFactored (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) :=
  ArrayProgramIO.Implementation.ofVariant (candidateVariant p r M dots scale jobs) (fun _ => True)
end
end VeriTile.Bench.Repairs.ContextAttnFwd.PairIO
