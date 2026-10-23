import bench.repairs.tritonbench_g.attention_kernel.Proofs.Context
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Complete source iteration, including both typed dots, tail mask, explicit
half conversions and all three pointer advances. -/
def bodyExec (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (hbn : 0<p.bn) (ctx : Context M p initial row a current) :
    {t : State α // run (engine (withDot M dot)) (parts p).body
      (current.setReg "start_n" .nat [] (fun _ => row))=some t} := by
  rcases p with ⟨storage,Q,K,V,Bias,Out,q,k,v,bias,o,batches,heads,length,dim,bm,bn,scale⟩
  cases storage
  all_goals
    have hquery:=ctx.query
    have hbias:=ctx.biasStatic
    simp only [TypedSource.Storage.float,TypedSource.Storage.tile,FloatDType.toTileDType,ofFloat,id_eq] at hquery hbias
    refine ⟨?_,?_⟩
    swap
    simp [parts,partsAs,kernel,kernelAs,TypedSource.repaired,TypedSource.FP16.repaired,TypedSource.BF16.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.startM,ctx.rows,ctx.biasOffset,ctx.outputOffset,hbias,hquery,ctx.keyPointer,ctx.valuePointer,
    ctx.maximum,ctx.normalizer,ctx.value,hbn]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
