import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Context
import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.Scatter

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite FusedRwkv6Kernel
open AttentionForwardTriton.Scatter (masked_scatter_memory masked_scatter_regs)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Every statement in the actual loop body executes, including the store
before its pointer advances. Successful execution is constructed, not assumed. -/
def bodyExec (M : Algebra α) (c : Shape) (initial current : State α) (scale : α)
    (n : Nat) (ctx : Context M c initial scale n current) :
    {t : State α // run (engine M) (parts c).body
      (current.setReg "_i" .nat [] (fun _ => n))=some t} := by
  by_cases hr : c.reverse=Bool.true
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
      Option.map,Option.getD,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralMod,numericLt,natLt,
      bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,hr,
      ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.km,ctx.vm,ctx.kvm,ctx.pq,ctx.pk,
      ctx.pw,ctx.pv,ctx.po,ctx.bias,ctx.state,ctx.scalar,FP.Structural.store,
      masked_scatter_regs]
    rfl

end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
