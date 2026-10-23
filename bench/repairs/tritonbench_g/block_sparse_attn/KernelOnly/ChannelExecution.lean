import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.Reference

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

structure Runtime {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (initial : State (Replicas α)) (carry : Carry α) (current : State (Replicas α)) : Prop
    extends Indices a p initial current where
  maximum : current.regs .real [16] "m_i" = some (fun i => carry.maximum i.1)
  denominator : current.regs .real [16] "l_i" = some (fun i => carry.denominator i.1)
  accumulator : current.regs .real [16,16] "acc" = some carry.output

structure ChannelInvariant {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (initial : State (Replicas α)) (carry : Carry α) (column : Int)
    (scores : Matrix α) (current : State (Replicas α)) : Prop
    extends Runtime a p initial carry current where
  start : current.regs .int [] "start_n" = some (fun _ => column*16)
  scores : current.regs .real [16,16] "qk" = some scores

def channelExecution {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (column : Int) (scores : Matrix α) (block : Nat)
    (h : ChannelInvariant a p initial carry column scores current) :
    {t : State (Replicas α) // run a.model (channelBody native p)
      (current.setReg "score_block" .nat [] (fun _ => block)) = some t} := by
  cases native <;> cases em : p.evenM <;> cases en : p.evenN
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [channelBody,compiledScoreBody,scoreBody,body,source,declared,em,en,
      carryNode,fp,ref2,ComputeKernel.surfaceBody,run,FP.Structural.step,
      evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,
      ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
      integralFloorDiv,integralMod,Arithmetic.model,
      h.scale,h.memory,h.qptrs,h.kptrs,h.rows,h.keys,h.length,h.start,h.scores,
      Broadcast.leftIndex,Broadcast.rightIndex]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
