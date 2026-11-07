import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardTransport
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardFinal
open VeriTile.Triton BackwardTransport
open ForwardTransport (store_observations)
noncomputable section
set_option maxHeartbeats 100000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def after (p : BackwardLayout) (length : Nat) : List Stmt :=
  (source p length).body.drop 10

theorem after_code (p : BackwardLayout) (length : Nat) :
    after p length =
    [.store .real [p.B] (.region p.gs (.castIntToNat (.ref .int [p.B] "col_offsets")))
      (.ref .real [p.B] "grad_s") (.mask (.ref .bool [p.B] "mask")),
     .store .real [p.B] (.region p.gl (.castIntToNat (.ref .int [p.B] "col_offsets")))
      (.ref .real [p.B] "grad_Lambda") (.mask (.ref .bool [p.B] "mask"))] := by
  unfold after source Erasure.backward TypedSource.backward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.backwardKernel
  simp only [
    ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody,ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.take,List.drop,List.map_cons,List.map_nil,ComputeStmt.listToAlgorithm?_map_alg,FloatDType.toTileDType]
  rfl

def sourceLast (p : BackwardLayout) (length : Nat) (base a : BlockState)
    (carried accumulated : Tile .real [p.B])
    (hc : a.regs .int [p.B] "col_offsets"=some (Tile.vec fun j=>(columns p base j : Int)))
    (hm : a.regs .bool [p.B] "mask"=some (mask p base))
    (hs : a.regs .real [p.B] "grad_s"=some carried)
    (hg : a.regs .real [p.B] "grad_Lambda"=some accumulated) :
    {last // stepStmts (after p length) a=some last} := by
  refine ⟨?_,?_⟩
  swap
  rw [after_code]
  simp only [stepStmts,stepStmt,↓eval_truncMod,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,
    NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,IntegralDType.floorDiv,
    ComparableDType.lt,Region.cast,bind,pure,Option.bind,Option.map,Option.pure_def,
    ↓BlockState.foldl_writeMemTyped_masked_regs,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    hc,hm,hs,hg,columns,mask,String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
  try rfl

def modelLast (p : BackwardLayout) (base a : BlockState)
    (carried accumulated : Tile .real [p.B])
    (hc : a.regs .nat [p.B] "col_offsets"=some (Tile.vec fun j=>columns p base j))
    (hm : a.regs .bool [p.B] "mask"=some (mask p base))
    (hs : a.regs .real [p.B] "grad_s"=some carried)
    (hg : a.regs .real [p.B] "grad_Lambda"=some accumulated) :
    {last // stepStmts (RealReference.diagSsmBackwardPostLoop p.gs p.gl p.B) a=some last} := by
  refine ⟨?_,?_⟩
  swap
  simp only [RealReference.diagSsmBackwardPostLoop]
  simp only [stepStmts,stepStmt,↓eval_truncMod,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,
    NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,IntegralDType.floorDiv,
    ComparableDType.lt,Region.cast,bind,pure,Option.bind,Option.map,Option.pure_def,
    ↓BlockState.foldl_writeMemTyped_masked_regs,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    hc,hm,hs,hg,columns,mask,String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
  try rfl

theorem last_related (p : BackwardLayout) (length : Nat) (base a b : BlockState)
    (carried accumulated : Tile .real [p.B]) (related : Related p base a b)
    (ha : a.regs .real [p.B] "grad_s"=some carried)
    (hb : b.regs .real [p.B] "grad_s"=some carried)
    (hga : a.regs .real [p.B] "grad_Lambda"=some accumulated)
    (hgb : b.regs .real [p.B] "grad_Lambda"=some accumulated) :
    (sourceLast p length base a carried accumulated related.sourceColumns related.sourceMask ha hga).val.ObsAgree
      (modelLast p base b carried accumulated related.modelColumns related.modelMask hb hgb).val := by
  simp only [sourceLast,modelLast,columns,mask,
    ↓BlockState.foldl_writeMemTyped_masked_regs,
    ↓BlockState.foldl_writeMemTyped_masked_regs,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,
    Tile.scalar,Tile.vec,Tile.bop,Tile.cop,Broadcast.leftIndex,Broadcast.rightIndex,
    Int.toNat_natCast,ha,hb,hga,hgb,related.sourceColumns,related.modelColumns,related.sourceMask,related.modelMask,
    String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
  apply store_observations
  exact store_observations _ _ _ _ _ _ _ related.observations

theorem source_code (p : BackwardLayout) (length : Nat) :
    (source p length).body = before p length ++
      [.forRangeInt "i" (.constInt 0) (.castNatToInt (.constNat length)) (.constNat 1) (body p length)] ++
        after p length := by
  unfold before body after source Erasure.backward TypedSource.backward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.backwardKernel
  simp only [
    ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody,ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.take,List.drop,List.map_cons,List.map_nil,ComputeStmt.listToAlgorithm?_map_alg,FloatDType.toTileDType]
  try rfl

theorem runs_related (p : BackwardLayout) (length : Nat) (u : BlockState) :
    ∃ a b, exec (source p length) u=some a ∧
      exec (RealReference.diag_ssm_backward_kernel p.s p.l p.y p.gs p.gx p.gl p.gy
        length p.batch p.dim p.B).toAlgKernel u=some b ∧ a.ObsAgree b := by
  let a0:=setup p length u
  let b0:=modelSetup p u
  have initial : Related p u a0.val b0.val := setup_related p length u
  obtain ⟨a,b,ha,hb,related⟩:=loop_related p u length 0 (Nat.zero_le _) a0.val b0.val initial
  simp only [Nat.cast_zero] at ha
  obtain ⟨carried,hca,hcb⟩:=related.carried
  obtain ⟨accumulated,hga,hgb⟩:=related.accumulated
  let lastA:=sourceLast p length u a carried accumulated related.sourceColumns related.sourceMask hca hga
  let lastB:=modelLast p u b carried accumulated related.modelColumns related.modelMask hcb hgb
  have observed : lastA.val.ObsAgree lastB.val := last_related p length u a b carried accumulated related hca hcb hga hgb
  refine ⟨lastA.val,lastB.val,?_,?_,observed⟩
  · unfold exec
    rw [source_code,ProgramRewrite.append,ProgramRewrite.append,a0.property]
    simp only [stepStmts,stepStmt,evalOp.eq_def,Tile.scalar,pure,bind,Option.bind_some,Int.ofNat_eq_natCast]
    rw [ha]
    simpa only [Option.bind_some] using lastA.property
  · unfold exec
    rw [RealReference.diag_ssm_backward_kernel_toAlg_body]
    simp only [RealReference.diagSsmBackwardProjectedBody,ProgramRewrite.append,b0.property,
      stepStmts,stepStmt,Option.bind_some,hb,lastB.property]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardFinal
