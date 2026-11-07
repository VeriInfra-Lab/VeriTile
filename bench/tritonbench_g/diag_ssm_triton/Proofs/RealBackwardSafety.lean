import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardFinal
import bench.tritonbench_g.diag_ssm_triton.Proofs.SignedRealLoop
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_BackwardIO
import VeriTile.Triton.Memory.TraceProofs

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.RealBackwardSafety
open VeriTile.Triton BackwardTransport
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem prefix_safe (p : BackwardLayout) (length : Nat) (u : BlockState) (bounds : RegionBounds)
    (hb : BackwardIO.Bounds p u length bounds) :
    Stmt.TraceSafeList bounds (before p length) u := by
  unfold before source Erasure.backward TypedSource.backward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.backwardKernel
  simp only [
    ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody,ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.take,List.drop,List.map_cons,List.map_nil,ComputeStmt.listToAlgorithm?_map_alg]
  simp only [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    and_true,true_and,and_self,stepStmts,stepStmt,↓eval_truncMod,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,
    NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,IntegralDType.floorDiv,
    ComparableDType.lt,Region.cast,bind,pure,Option.bind,Option.map,Option.pure_def,
    ↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false]
  simp only [Int.ofNat_eq_natCast,←Int.natCast_mul,←Int.natCast_add,←Int.ofNat_tmod,Int.toNat_natCast,Int.ofNat_lt]
  simp only [Option.some.injEq,and_true,true_and]
  intro offsets hoff j ⟨masks,hm,hact⟩
  subst offsets; subst masks
  exact hb.transition j.1 (by change _ < _; exact of_decide_eq_true hact)

theorem previous_address (time D c : Nat) (positive : 0<time) :
    time*D+c-D=(time-1)*D+c := by
  have ht : time=(time-1)+1 := by omega
  have hm : time*D=(time-1)*D+D := by
    calc time*D=((time-1)+1)*D := by rw [←ht]
         _=(time-1)*D+D := by simp only [Nat.add_mul,Nat.one_mul]
  omega

theorem body_safe (p : BackwardLayout) (base : BlockState) (length n : Nat) (bound : n<length)
    (a b : BlockState) (related : Related p base a b) (bounds : RegionBounds)
    (hb : BackwardIO.Bounds p base length bounds) :
    Stmt.TraceSafeList bounds (body p length) (a.setReg "i" .int [] (Tile.scalar (n : Int))) := by
  obtain ⟨carried,hs,_⟩:=related.carried
  obtain ⟨accumulated,hg,_⟩:=related.accumulated
  obtain ⟨transition,hl,_⟩:=related.transition
  have hc:=related.sourceColumns
  have hm:=related.sourceMask
  unfold body source Erasure.backward TypedSource.backward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.backwardKernel
  simp only [
    ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody,ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.take,List.drop,List.map_cons,List.map_nil,ComputeStmt.listToAlgorithm?_map_alg]
  have indexcast : (length : Int)-1-n=((length-1-n : Nat) : Int) := by omega
  have compare : ((length-1-n : Nat) : Int)>0 ↔ length-1-n>0 := by omega
  by_cases branch : length-1-n>0
  all_goals
  {
    simp only [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
      MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      and_true,true_and,and_self,stepStmts,stepStmt,↓eval_truncMod,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
      Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,
      NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,IntegralDType.floorDiv,
      ComparableDType.lt,ComparableDType.gt,Region.cast,bind,pure,Option.bind,Option.map,Option.pure_def,
      ↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
      BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
      hc,hm,hs,hg,hl,columns,mask,String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false]
    simp only [Int.ofNat_eq_natCast,indexcast,compare,branch,decide_true,decide_false,
      Bool.false_eq_true,if_false,if_true,Option.bind,bind,pure,Option.pure_def,
      ↓BlockState.setReg_same,↓BlockState.setReg_ne_name,hc,hm,hs,hg,hl,columns,mask,
      BlockState.setReg_readMem,BlockState.setReg_readMemValue,BlockState.readMemValue_real,
      String.reduceEq,ne_eq,not_false_eq_true,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.toNat_sub',Nat.mul_assoc]
    try simp only [previous_address,branch]
    simp only [Option.some.injEq,and_true,true_and]
    refine ⟨?_,?_,?_⟩
    all_goals intro offsets hoff j ⟨masks,hmask,hact⟩; subst offsets; subst masks
    all_goals first
      | exact hb.gradOutput (length-1-n) (by omega) j.1 (by change _ < _; exact of_decide_eq_true hact)
      | exact hb.gradInput (length-1-n) (by omega) j.1 (by change _ < _; exact of_decide_eq_true hact)
      | exact hb.initial (by omega) j.1 (by change _ < _; exact of_decide_eq_true hact)
      | exact hb.previous (length-1-n-1) (by omega) j.1 (by change _ < _; exact of_decide_eq_true hact)
  }

theorem post_safe (p : BackwardLayout) (length : Nat) (base a b : BlockState)
    (related : Related p base a b) (bounds : RegionBounds)
    (hb : BackwardIO.Bounds p base length bounds) :
    Stmt.TraceSafeList bounds (BackwardFinal.after p length) a := by
  obtain ⟨carried,hs,_⟩:=related.carried
  obtain ⟨accumulated,hg,_⟩:=related.accumulated
  have hc:=related.sourceColumns
  have hm:=related.sourceMask
  rw [BackwardFinal.after_code]
  simp only [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    and_true,true_and,and_self,stepStmts,stepStmt,↓eval_truncMod,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,
    NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,IntegralDType.floorDiv,
    ComparableDType.lt,ComparableDType.gt,Region.cast,bind,pure,Option.bind,Option.map,Option.pure_def,
    ↓BlockState.foldl_writeMemTyped_masked_regs,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    hc,hm,hs,hg,columns,mask,String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false]
  simp only [Int.toNat_natCast,Option.some.injEq]
  constructor
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hb.gradInitial j.1 (by change _ < _; exact of_decide_eq_true hact)
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hb.gradTransition j.1 (by change _ < _; exact of_decide_eq_true hact)

theorem trace_safe (p : BackwardLayout) (length : Nat) (u : BlockState) (bounds : RegionBounds)
    (hb : BackwardIO.Bounds p u length bounds) : (source p length).TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  rw [BackwardFinal.source_code,List.append_assoc]
  apply TraceProofs.append_safe _ u (prefix_safe p length u bounds hb)
  intro a he
  have arun := (setup p length u).property
  have same : a=(setup p length u).val := Option.some.inj (he.symm.trans arun)
  subst a
  let P := fun a=>∃ b, Related p u a b
  have initial : P (setup p length u).val := ⟨_,setup_related p length u⟩
  have preserve : ∀ n a next, n<length → P a →
      stepStmts (body p length) (a.setReg "i" .int [] (Tile.scalar (n : Int)))=some next → P next := by
    intro n a next bound ha he
    obtain ⟨b,related⟩:=ha
    obtain ⟨carried,hca,hcb⟩:=related.carried
    obtain ⟨accumulated,hga,hgb⟩:=related.accumulated
    obtain ⟨transition,hla,hlb⟩:=related.transition
    let sa:=sourceNext p u length n bound a carried accumulated transition related.sourceColumns related.sourceMask hca hga hla
    let sb:=modelNext p u length n bound b carried accumulated transition related.modelColumns related.modelMask hcb hgb hlb
    have observed := next_related p u length n bound a b carried accumulated transition related hca hcb hga hgb hla hlb
    have same : next=sa.val := Option.some.inj (he.symm.trans sa.property)
    subst next
    exact ⟨sb.val,observed⟩
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe,Op.SafeAt.eq_def,true_and,evalOp.eq_def,Tile.scalar,pure,bind,Option.bind_some,Int.ofNat_eq_natCast]
    have hs := VeriTile.Bench.Repairs.DiagSsmTriton.SignedRealLoop.safe bounds "i" length (body p length) P
      (fun n a bound ha=>by obtain ⟨b,related⟩:=ha; exact body_safe p u length n bound a b related bounds hb)
      preserve 0 (Nat.zero_le _) _ initial
    simpa only [Nat.cast_zero] using hs
  · intro next he
    obtain ⟨a,ha,hP⟩ := VeriTile.Bench.Repairs.DiagSsmTriton.SignedRealLoop.invariant "i" length
      (body p length) (fun _=>P)
      (fun n a bound h=>by
        obtain ⟨b,related⟩:=h
        obtain ⟨carried,hc,_⟩:=related.carried
        obtain ⟨accumulated,hg,_⟩:=related.accumulated
        obtain ⟨transition,hl,_⟩:=related.transition
        let sa:=sourceNext p u length n bound a carried accumulated transition related.sourceColumns related.sourceMask hc hg hl
        exact ⟨sa.val,sa.property,preserve n a sa.val bound ⟨b,related⟩ sa.property⟩)
      0 (Nat.zero_le _) _ initial
    simp only [Nat.cast_zero] at ha
    simp only [stepStmt,evalOp.eq_def,Tile.scalar,pure,bind,Option.bind_some,Int.ofNat_eq_natCast,ha] at he
    have same : a=next := Option.some.inj he
    subst next
    obtain ⟨b,related⟩:=hP
    exact post_safe p length u a b related bounds hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.RealBackwardSafety
