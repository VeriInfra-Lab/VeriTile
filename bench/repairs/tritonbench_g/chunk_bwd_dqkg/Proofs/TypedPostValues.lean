import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedAnswer
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def contextPost (M : Algebra α) (dot : Dot α) (c : Config)
    (initial current : State α) (a : Carries α c)
    (ctx : Context M c initial a current) :
    {t : State α // run (engine (withDot M dot)) ((parts c).after.take 17) current=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    String.reduceEq,beq_self_eq_true,↓reduceIte,List.take,List.drop]
  simp [run,step.eq_2,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,
    Option.bind,engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,
    ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,natLe,numericLe,
    numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,TileShape.product,Region.cast,
    State.setReg,ctx.memory,ctx.key,ctx.time,ctx.head,ctx.heads,ctx.rows,ctx.gate,ctx.last,ctx.dq,ctx.dk,ctx.ds,ctx.dgLast,ctx.dg]
  rfl

private theorem min_lt (a b : Nat) : min a b=(if a<b then a else b) := by split_ifs <;> omega

theorem contextPost_context (M : Algebra α) (dot : Dot α) (c : Config)
    (initial current : State α) (a : Carries α c)
    (ctx : Context M c initial a current) :
    AnswerContext M dot c initial a (contextPost M dot c initial current a ctx).val := by
  constructor
  all_goals simp [contextPost,ctx.memory,ctx.key,ctx.time,ctx.head,ctx.heads,ctx.rows,ctx.gate,ctx.last,ctx.dq,ctx.dk,ctx.ds,ctx.dgLast,ctx.dg,
    answerQ,answerK,answerG,gateContribution,lastCorrection,gatedQ,gatedK,scores,
    keyTile,qPointer,gPointer,totalProduct,flatProduct,add,mul,sub,exp,zero,
    bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,TileShape.indexToList,
    min_lt]
  all_goals try funext i
  all_goals try simp [answerQ,answerK,answerG,gateContribution,lastCorrection,gatedQ,gatedK,scores,
    keyTile,qPointer,gPointer,totalProduct,flatProduct,add,mul,sub,exp,zero,
    bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,TileShape.indexToList,
    min_lt]
  all_goals try norm_num only
  all_goals try rfl

theorem post_context (M : Algebra α) (dot : Dot α) (c : Config)
    (initial current : State α) (a : Carries α c)
    (ctx : Context M c initial a current) (ready : Ready c current) :
    AnswerContext M dot c initial a (post M dot c current ready).val := by
  have eq := Option.some.inj ((post M dot c current ready).property.symm.trans
    (contextPost M dot c initial current a ctx).property)
  rw [eq]
  exact contextPost_context M dot c initial current a ctx

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
