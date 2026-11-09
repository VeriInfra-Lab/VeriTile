import bench.tritonbench_g.rms_matmul_rbe.Proofs.PrefixValues

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- The actual signed source bound is exactly ceiling(K/64), including K=0.
Only discrete arithmetic is used; the floating algebra is arbitrary. -/
theorem range_header {α : Type} [Inhabited α] (M : Algebra α)
    (p : Arguments) (s : State α) :
    evalOp M none (startExpr p) s=some (fun _=>0) ∧
    evalOp M none (countExpr p) s=some (fun _=>(((p.K+63)/64:Nat):Int)) ∧
    evalOp M none (strideExpr p) s=some (fun _=>1) := by
  simp (config := {maxSteps:=20000}) [startExpr,countExpr,strideExpr,source,
    TypedSource.kernel,TypedSource.halfHalf,TypedSource.rmsKernel,ComputeKernel.surfaceBody,
    evalOp_unfold,Option.bind,bop,numeric,numericEq,natLt,numericLt,integralFloorDiv,integralMod,
    Broadcast.leftIndex,Broadcast.rightIndex]
  funext i
  cases i
  change (p.K:Int)/64+(if (p.K:Int)-64*((p.K:Int)/64)=0 then 0 else 1)=
    ((p.K:Int)+63)/64
  split_ifs <;> omega

#axiomsClean range_header
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
