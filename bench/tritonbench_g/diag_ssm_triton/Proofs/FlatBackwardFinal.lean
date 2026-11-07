import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardFinal
import bench.tritonbench_g.diag_ssm_triton.Proofs.FlatBackwardTransport
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.FlatBackwardFinal
open VeriTile.Triton FlatBackwardTransport
open BackwardTransport (Related columns mask)
open ForwardTransport (store_observations)
noncomputable section
set_option maxHeartbeats 100000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def after (A : FlatAlloc) (p : BackwardLayout) (length : Nat) : List Stmt :=
  A.flattenStmts (BackwardFinal.after p length)

def sourceLast (A : FlatAlloc) (p : BackwardLayout) (length : Nat) (base a : BlockState)
    (carried accumulated : Tile .real [p.B])
    (hc : a.regs .int [p.B] "col_offsets"=some (Tile.vec fun j=>(columns p base j : Int)))
    (hm : a.regs .bool [p.B] "mask"=some (mask p base))
    (hs : a.regs .real [p.B] "grad_s"=some carried)
    (hg : a.regs .real [p.B] "grad_Lambda"=some accumulated) :
    {last // stepStmts (after A p length) a=some last} := by
  refine ⟨?_,?_⟩
  swap
  unfold after
  rw [BackwardFinal.after_code]
  simp only [FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast,stepStmts,stepStmt,↓eval_truncMod,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,
    NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,IntegralDType.floorDiv,
    ComparableDType.lt,Region.cast,bind,pure,Option.bind,Option.map,Option.pure_def,
    ↓BlockState.foldl_writeMemTyped_masked_regs,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    hc,hm,hs,hg,columns,mask,String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
  try rfl

def modelLast (A : FlatAlloc) (p : BackwardLayout) (base a : BlockState)
    (carried accumulated : Tile .real [p.B])
    (hc : a.regs .nat [p.B] "col_offsets"=some (Tile.vec fun j=>columns p base j))
    (hm : a.regs .bool [p.B] "mask"=some (mask p base))
    (hs : a.regs .real [p.B] "grad_s"=some carried)
    (hg : a.regs .real [p.B] "grad_Lambda"=some accumulated) :
    {last // stepStmts (A.flattenStmts (RealReference.diagSsmBackwardPostLoop p.gs p.gl p.B)) a=some last} := by
  refine ⟨?_,?_⟩
  swap
  simp only [RealReference.diagSsmBackwardPostLoop,FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast]
  simp only [FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast,stepStmts,stepStmt,↓eval_truncMod,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,
    NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,IntegralDType.floorDiv,
    ComparableDType.lt,Region.cast,bind,pure,Option.bind,Option.map,Option.pure_def,
    ↓BlockState.foldl_writeMemTyped_masked_regs,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    hc,hm,hs,hg,columns,mask,String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
  try rfl

theorem last_related (A : FlatAlloc) (p : BackwardLayout) (length : Nat) (base a b : BlockState)
    (carried accumulated : Tile .real [p.B]) (related : Related p base a b)
    (ha : a.regs .real [p.B] "grad_s"=some carried)
    (hb : b.regs .real [p.B] "grad_s"=some carried)
    (hga : a.regs .real [p.B] "grad_Lambda"=some accumulated)
    (hgb : b.regs .real [p.B] "grad_Lambda"=some accumulated) :
    (sourceLast A p length base a carried accumulated related.sourceColumns related.sourceMask ha hga).val.ObsAgree
      (modelLast A p base b carried accumulated related.modelColumns related.modelMask hb hgb).val := by
  simp only [sourceLast,modelLast,columns,mask,
    ↓BlockState.foldl_writeMemTyped_masked_regs,
    ↓BlockState.foldl_writeMemTyped_masked_regs,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,
    Tile.scalar,Tile.vec,Tile.bop,Tile.cop,Broadcast.leftIndex,Broadcast.rightIndex,
    Int.toNat_natCast,ha,hb,hga,hgb,related.sourceColumns,related.modelColumns,related.sourceMask,related.modelMask,
    String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
  apply store_observations
  exact store_observations _ _ _ _ _ _ _ related.observations

theorem source_code (A : FlatAlloc) (p : BackwardLayout) (length : Nat) :
    (source A p length).body = before A p length ++
      [.forRangeInt "i" (.constInt 0) (.castNatToInt (.constNat length)) (.constNat 1) (body A p length)] ++
        after A p length := by
  unfold before body after BackwardFinal.after BackwardTransport.source source Erasure.backward TypedSource.backward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.backwardKernel
  simp only [
    ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody,ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.take,List.drop,List.map_cons,List.map_nil,ComputeStmt.listToAlgorithm?_map_alg,FloatDType.toTileDType,FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast]
  try rfl

theorem runs_related (A : FlatAlloc) (p : BackwardLayout) (length : Nat) (u : BlockState) :
    ∃ a b, exec (source A p length) u=some a ∧
      exec (A.flattenKernel (RealReference.diag_ssm_backward_kernel p.s p.l p.y p.gs p.gx p.gl p.gy
        length p.batch p.dim p.B).toAlgKernel) u=some b ∧ a.ObsAgree b := by
  let a0:=setup A p length u
  let b0:=modelSetup A p u
  have initial : Related p u a0.val b0.val := setup_related A p length u
  obtain ⟨a,b,ha,hb,related⟩:=loop_related A p u length 0 (Nat.zero_le _) a0.val b0.val initial
  simp only [Nat.cast_zero] at ha
  obtain ⟨carried,hca,hcb⟩:=related.carried
  obtain ⟨accumulated,hga,hgb⟩:=related.accumulated
  let lastA:=sourceLast A p length u a carried accumulated related.sourceColumns related.sourceMask hca hga
  let lastB:=modelLast A p u b carried accumulated related.modelColumns related.modelMask hcb hgb
  have observed : lastA.val.ObsAgree lastB.val := last_related A p length u a b carried accumulated related hca hcb hga hgb
  refine ⟨lastA.val,lastB.val,?_,?_,observed⟩
  · unfold exec
    rw [source_code,ProgramRewrite.append,ProgramRewrite.append,a0.property]
    simp only [FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast,stepStmts,stepStmt,evalOp.eq_def,Tile.scalar,pure,bind,Option.bind_some,Int.ofNat_eq_natCast]
    rw [ha]
    simpa only [Option.bind_some] using lastA.property
  · unfold exec
    change stepStmts (A.flattenStmts (RealReference.diag_ssm_backward_kernel p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).toAlgKernel.body) u=some lastB.val
    rw [RealReference.diag_ssm_backward_kernel_toAlg_body]
    simp only [RealReference.diagSsmBackwardProjectedBody,FlatAlloc.flattenStmts_append,FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast,ProgramRewrite.append,b0.property,
      stepStmts,stepStmt,Option.bind_some,hb,lastB.property]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.FlatBackwardFinal
