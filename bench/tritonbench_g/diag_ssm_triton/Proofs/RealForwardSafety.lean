import bench.tritonbench_g.diag_ssm_triton.Proofs.ForwardTransport
import bench.tritonbench_g.diag_ssm_triton.Proofs.SignedRealLoop
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ForwardIO
import VeriTile.Triton.Memory.TraceProofs

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.RealForwardSafety
open VeriTile.Triton ForwardTransport
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem prefix_safe (p : ForwardLayout) (length : Nat) (u : BlockState) (bounds : RegionBounds)
    (hb : ForwardIO.Bounds p u length bounds) :
    Stmt.TraceSafeList bounds (before p length) u := by
  unfold before source Erasure.forward TypedSource.forward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.forwardKernel
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
  simp only [Option.some.injEq]
  constructor
  · intro offsets hoff j ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.initial j.1 (by change _ < _; exact of_decide_eq_true hact)
  · intro offsets hoff j ⟨masks,hm,hact⟩
    subst offsets; subst masks
    exact hb.transition j.1 (by change _ < _; exact of_decide_eq_true hact)

theorem body_safe (p : ForwardLayout) (base : BlockState) (length n : Nat) (bound : n<length)
    (a b : BlockState) (related : Related p base a b) (bounds : RegionBounds)
    (hb : ForwardIO.Bounds p base length bounds) :
    Stmt.TraceSafeList bounds (body p) (a.setReg "t" .int [] (Tile.scalar (n : Int))) := by
  obtain ⟨carried,hs,_⟩:=related.carried
  obtain ⟨transition,hl,_⟩:=related.transition
  have hc:=related.sourceColumns
  have hm:=related.sourceMask
  unfold body source Erasure.forward TypedSource.forward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.forwardKernel
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
    hc,hm,hs,hl,columns,mask,String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false]
  simp only [Int.ofNat_eq_natCast,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.mul_assoc]
  simp only [Option.some.injEq]
  constructor
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hb.input n bound j.1 (by change _ < _; exact of_decide_eq_true hact)
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hb.output n bound j.1 (by change _ < _; exact of_decide_eq_true hact)

theorem trace_safe (p : ForwardLayout) (length : Nat) (u : BlockState) (bounds : RegionBounds)
    (hb : ForwardIO.Bounds p u length bounds) : (source p length).TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  rw [source_code]
  apply TraceProofs.append_safe _ u (prefix_safe p length u bounds hb)
  intro a he
  have arun := (setup p length u).property
  have same : a=(setup p length u).val := Option.some.inj (he.symm.trans arun)
  subst a
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  simp only [Stmt.TraceSafe,Op.SafeAt.eq_def,true_and,evalOp.eq_def,Tile.scalar,pure,bind,Option.bind_some,Int.ofNat_eq_natCast]
  let P := fun a=>∃ b, Related p u a b
  have hs := VeriTile.Bench.Repairs.DiagSsmTriton.SignedRealLoop.safe bounds "t" length (body p) P ?_ ?_
    0 (Nat.zero_le _) _ ⟨_,setup_related p length u⟩
  · simpa only [Nat.cast_zero] using hs
  · intro n a bound ha
    obtain ⟨b,related⟩:=ha
    exact body_safe p u length n bound a b related bounds hb
  · intro n a next bound ha he
    obtain ⟨b,related⟩:=ha
    obtain ⟨carried,hca,hcb⟩:=related.carried
    obtain ⟨transition,hla,hlb⟩:=related.transition
    let sa:=sourceNext p u n a carried transition related.sourceColumns related.sourceMask hca hla
    let sb:=modelNext p u n b carried transition related.modelColumns related.modelMask hcb hlb
    have observed := next_related p u n a b carried transition related hca hcb hla hlb
    have same : next=sa.val := Option.some.inj (he.symm.trans sa.property)
    subst next
    exact ⟨sb.val,observed⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.RealForwardSafety
