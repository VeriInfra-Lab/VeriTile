import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Values
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def stop (p : Parameters) (s : State α) : Nat := min ((s.pids 0+1)*128) p.length

def middleExec (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    {t : State α // run (engine (withDot M dot)) (middle p) current=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [middle,parts,kernel,TypedSource.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.startM,ctx.keyBase,ctx.valueBase,ctx.scaleBase]
  rfl

theorem middle_context (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    Context M p initial (initial.pids 0*128) a (middleExec M dot p initial current row a ctx).val := by
  constructor
  all_goals try simp [middleExec,ctx.memory,ctx.startM,ctx.firstLo,ctx.firstHi,ctx.keyBase,ctx.valueBase,ctx.scaleBase,
    ctx.rows,ctx.cols,ctx.query,ctx.scale,ctx.outputPointer,ctx.maximum,ctx.normalizer,ctx.value,
    Parameters.q,Parameters.k,Parameters.v,Parameters.o,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals try funext i
  all_goals try simp [Parameters.q,Parameters.k,Parameters.v,Parameters.o,
    Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.mul_mod,Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.mul_comm,Nat.mul_left_comm,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

  all_goals ring

#axiomsClean middle_context
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Source
