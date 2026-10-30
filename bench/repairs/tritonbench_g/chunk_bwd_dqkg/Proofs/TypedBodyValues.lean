import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedValues
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem chosen_eq {d : TileDType} {sh : TileShape} {s : State α} {n : RegName}
    (h : ∃ v : Values α d sh,s.regs d sh n=some v) (v : Values α d sh)
    (hv : s.regs d sh n=some v) : Classical.choose h=v :=
  Option.some.inj ((Classical.choose_spec h).symm.trans hv)

def contextBody (M : Algebra α) (dot : Dot α) (c : Config)
    (initial current : State α) (block : Nat) (a : Carries α c)
    (ctx : Context M c initial a current) :
    {t : State α // run (engine (withDot M dot)) (parts c).body
      (current.setReg "i_v" .nat [] (fun _=>block))=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    run,step.eq_2,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,
    Option.bind,engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,
    ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,natLe,numericLe,
    numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,TileShape.product,Region.cast,
    State.setReg,ctx.memory,ctx.key,ctx.time,ctx.head,ctx.heads,ctx.rows,ctx.gate,ctx.last,ctx.dq,ctx.dk,ctx.ds,ctx.dgLast,ctx.dg]
  rfl

theorem contextBody_context (M : Algebra α) (dot : Dot α) (c : Config)
    (initial current : State α) (block : Nat) (a : Carries α c)
    (ctx : Context M c initial a current) :
    Context M c initial (updateCarries M dot c initial block a)
      (contextBody M dot c initial current block a ctx).val := by
  constructor
  all_goals simp [contextBody,ctx.memory,ctx.key,ctx.time,ctx.head,ctx.heads,ctx.rows,ctx.gate,ctx.last,ctx.dq,ctx.dk,ctx.ds,ctx.dgLast,ctx.dg,
    updateCarries,valTile,histTile,totalProduct,flatProduct,add,mul,zero,
    bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,TileShape.indexToList]
  all_goals try funext i
  all_goals try simp [updateCarries,valTile,histTile,totalProduct,flatProduct,add,mul,zero,
    bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,TileShape.indexToList]
  all_goals try rfl

theorem body_context (M : Algebra α) (dot : Dot α) (c : Config)
    (initial current : State α) (block : Nat) (a : Carries α c)
    (ctx : Context M c initial a current) (ready : Ready c current) :
    Context M c initial (updateCarries M dot c initial block a)
      (body M dot c current ready block) := by
  have eq := Option.some.inj ((body_runs M dot c current ready block).symm.trans
    (contextBody M dot c initial current block a ctx).property)
  rw [eq]
  exact contextBody_context M dot c initial current block a ctx

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
