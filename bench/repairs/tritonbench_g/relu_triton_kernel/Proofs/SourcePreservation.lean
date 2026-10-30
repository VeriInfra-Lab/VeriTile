import bench.repairs.tritonbench_g.relu_triton_kernel.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ReluTritonKernel.SourcePreservation
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- The raw source and the existing physical model in element coordinates
have identical complete optional states for every backend, including failure.
Only natural address identities 0+1*i=i are used; no FP cast law is assumed. -/
theorem original {A : Type} [Inhabited A] (M : Algebra A)
    (input output : RegionName) (N B : Nat) (s : State A) :
    exec M (body input output N B) s =
      exec M (kernel input output 0 0 1 N B) s := by
  cases hcmp : M.compareLe (some .fp32) .real <;>
  simp [hcmp,store,ComputeDType.eraseDType,ofFloat,toFloat,body,kernel,Coordinates.kernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,numericLe,numericLt,
    natLt,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

theorem candidate {A : Type} [Inhabited A] (M : Algebra A)
    (input output : RegionName) (N B : Nat) (s : State A) :
    exec M (inlinedBody input output N B) s =
      exec M (fusedKernel input output 0 0 1 N B) s := by
  cases hcmp : M.compareLe (some .fp32) .real <;>
  simp [hcmp,store,ComputeDType.eraseDType,ofFloat,toFloat,inlinedBody,fusedKernel,Coordinates.fusedKernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,numericLe,numericLt,
    natLt,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

#axiomsClean original
#axiomsClean candidate
end
end VeriTile.Bench.Repairs.ReluTritonKernel.SourcePreservation
