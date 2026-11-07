import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardTransport
import bench.tritonbench_g.diag_ssm_triton.Proofs.ForwardTransport
import VeriTile.Triton.Semantics.Integer
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_RealReference
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_Erasure
import VeriTile.Triton.Memory.FlattenTolerant

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.FlatBackwardTransport
open VeriTile.Triton
open BackwardTransport (Related columns mask)
open ForwardTransport (store_observations)

noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev source (A : FlatAlloc) (p : BackwardLayout) (length : Nat) :=
  A.flattenKernel (Erasure.backward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).toAlgKernel

def before (A : FlatAlloc) (p : BackwardLayout) (length : Nat) : List Stmt :=
  (source A p length).body.take 9

def body (A : FlatAlloc) (p : BackwardLayout) (length : Nat) : List Stmt :=
  match (source A p length).body.drop 9 with
  | .forRangeInt _ _ _ _ loop :: _ => loop
  | _ => []

/-- The witness is obtained by running the complete actual signed prefix. -/
def setup (A : FlatAlloc) (p : BackwardLayout) (length : Nat) (u : BlockState) :
    {a // stepStmts (before A p length) u=some a} := by
  refine ⟨?_,?_⟩
  swap
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
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.take,List.drop,List.map_cons,List.map_nil,ComputeStmt.listToAlgorithm?_map_alg,FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast]
  simp only [FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast,stepStmts,stepStmt,↓eval_truncMod,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,
    NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,IntegralDType.floorDiv,
    ComparableDType.lt,Region.cast,bind,pure,Option.bind,Option.map,Option.pure_def,
    ↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
  try rfl


def modelSetup (A : FlatAlloc) (p : BackwardLayout) (u : BlockState) :
    {b // stepStmts (A.flattenStmts (RealReference.diagSsmBackwardPreLoop p.l p.batch p.dim p.B)) u=some b} := by
  refine ⟨?_,?_⟩
  swap
  simp only [RealReference.diagSsmBackwardPreLoop,FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast,stepStmts,stepStmt,evalOp.eq_def,
    Tile.scalar,Tile.vec,Tile.bop,Tile.cop,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.mul,NumericDType.add,IntegralDType.mod,ComparableDType.lt,Region.cast,
    bind,pure,Option.bind,Option.map,Option.pure_def,
    ↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
  try rfl

theorem setup_related (A : FlatAlloc) (p : BackwardLayout) (length : Nat) (u : BlockState) :
    Related p u (setup A p length u).val (modelSetup A p u).val := by
  constructor
  · exact ⟨rfl,rfl,rfl,rfl⟩
  all_goals
    simp only [setup,modelSetup,columns,mask,BlockState.ObsAgree,
      ↓BlockState.setReg_same,↓BlockState.setReg_ne_name,
      BlockState.setReg_mem,BlockState.setReg_pids,BlockState.setReg_numPids,
      BlockState.setReg_undef,Tile.vec,Tile.scalar,Tile.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,
      Int.ofNat_eq_natCast,←Int.natCast_add,←Int.natCast_mul,←Int.ofNat_tmod,
      Int.toNat_natCast,Int.ofNat_lt,
      String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,
      Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
  all_goals first | rfl | exact ⟨_,rfl,rfl⟩




def sourceNext (A : FlatAlloc) (p : BackwardLayout) (base : BlockState) (length n : Nat) (bound : n<length) (a : BlockState)
    (carried accumulated transition : Tile .real [p.B])
    (hc : a.regs .int [p.B] "col_offsets"=
      some (Tile.vec fun j=>((columns p base j : Nat) : Int)))
    (hm : a.regs .bool [p.B] "mask"=some (mask p base))
    (hs : a.regs .real [p.B] "grad_s"=some carried)
    (hg : a.regs .real [p.B] "grad_Lambda"=some accumulated)
    (hl : a.regs .real [p.B] "Lambda"=some transition) :
    {next // stepStmts (body A p length) (a.setReg "i" .int [] (Tile.scalar (n : Int)))=some next} := by
  have compare : (0 : Int)<(length : Int)-1-n ↔ (n : Int)<(length : Int)-1 := by omega
  by_cases branch : (n : Int)<(length : Int)-1
  all_goals
  {
    refine ⟨?_,?_⟩
    swap
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
      ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.take,List.drop,List.map_cons,List.map_nil,ComputeStmt.listToAlgorithm?_map_alg,FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast]
    simp only [FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast,Int.ofNat_eq_natCast,compare,stepStmts,stepStmt,↓eval_truncMod,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
      Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,
      NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,IntegralDType.floorDiv,
      ComparableDType.lt,ComparableDType.gt,Region.cast,bind,pure,Option.bind,Option.map,Option.pure_def,
      branch,hc,hm,hs,hg,hl,columns,mask,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
      BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
      String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
    try rfl
  }

def modelNext (A : FlatAlloc) (p : BackwardLayout) (base : BlockState) (length n : Nat) (bound : n<length) (a : BlockState)
    (carried accumulated transition : Tile .real [p.B])
    (hc : a.regs .nat [p.B] "col_offsets"=some (Tile.vec fun j=>columns p base j))
    (hm : a.regs .bool [p.B] "mask"=some (mask p base))
    (hs : a.regs .real [p.B] "grad_s"=some carried)
    (hg : a.regs .real [p.B] "grad_Lambda"=some accumulated)
    (hl : a.regs .real [p.B] "Lambda"=some transition) :
    {next // stepStmts (A.flattenStmts (RealReference.diagSsmBackwardLoopBody p.s p.y p.gx p.gy length p.batch p.dim p.B))
      (a.setReg "i" .nat [] (Tile.scalar n))=some next} := by
  by_cases branch : length-1-n>0
  all_goals
  {
    refine ⟨?_,?_⟩
    swap
    simp only [RealReference.diagSsmBackwardLoopBody,FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast,
      stepStmts,stepStmt,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
      Broadcast.leftIndex,Broadcast.rightIndex,
      NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,ComparableDType.lt,ComparableDType.gt,Region.cast,
      bind,pure,Option.bind,Option.map,Option.pure_def,
      branch,hc,hm,hs,hg,hl,columns,mask,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,
      BlockState.setReg_pids,BlockState.setReg_readMemValue,BlockState.setReg_readMem,
      BlockState.readMemValue_real,
      String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
    try rfl
  }

theorem next_related (A : FlatAlloc) (p : BackwardLayout) (base : BlockState) (length n : Nat) (bound : n<length) (a b : BlockState)
    (carried accumulated transition : Tile .real [p.B]) (related : Related p base a b)
    (ha : a.regs .real [p.B] "grad_s"=some carried)
    (hb : b.regs .real [p.B] "grad_s"=some carried)
    (hga : a.regs .real [p.B] "grad_Lambda"=some accumulated)
    (hgb : b.regs .real [p.B] "grad_Lambda"=some accumulated)
    (hla : a.regs .real [p.B] "Lambda"=some transition)
    (hlb : b.regs .real [p.B] "Lambda"=some transition) :
    Related p base
      (sourceNext A p base length n bound a carried accumulated transition related.sourceColumns related.sourceMask ha hga hla).val
      (modelNext A p base length n bound b carried accumulated transition related.modelColumns related.modelMask hb hgb hlb).val := by
  have indexcast : (length : Int)-1-n=((length-1-n : Nat) : Int) := by omega
  by_cases branch : length-1-n>0
  all_goals have flag : ((n : Int)<(length : Int)-1) ↔ (length-1-n>0) := by omega
  all_goals constructor
  all_goals
    simp only [sourceNext,modelNext,flag,branch,dif_pos,dif_neg,columns,mask,
      ↓BlockState.foldl_writeMemTyped_masked_regs,
      ↓BlockState.setReg_same,↓BlockState.setReg_ne_name,
      Tile.scalar,Tile.vec,Tile.bop,Tile.cop,Broadcast.leftIndex,Broadcast.rightIndex,
      Int.ofNat_eq_natCast,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.toNat_sub',indexcast,
      Nat.mul_assoc,related.observations.readMem_eq,
      ha,hb,hga,hgb,hla,hlb,related.sourceColumns,related.modelColumns,related.sourceMask,related.modelMask,
      String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false,decide_true,decide_false]
  all_goals first
    | exact store_observations _ _ _ _ _ _ _ related.observations
    | rfl
    | exact ⟨_,rfl,rfl⟩



theorem loop_related (A : FlatAlloc) (p : BackwardLayout) (base : BlockState) (length start : Nat)
    (bound : start≤length) (a b : BlockState) (related : Related p base a b) :
    ∃ x y,
      stepForRangeIntAux "i" (start : Int) (length : Int) 1 (body A p length) a=some x ∧
      stepForRangeAux "i" start length 1
        (A.flattenStmts (RealReference.diagSsmBackwardLoopBody p.s p.y p.gx p.gy length p.batch p.dim p.B)) b=some y ∧
      Related p base x y := by
  by_cases active : start<length
  · have activeInt : (start : Int)<length := by exact_mod_cast active
    rw [stepForRangeIntAux.step_lt (by decide) activeInt,stepForRangeAux.step_lt (by decide) active]
    obtain ⟨carried,ha,hb⟩:=related.carried
    obtain ⟨accumulated,hga,hgb⟩:=related.accumulated
    obtain ⟨transition,hla,hlb⟩:=related.transition
    let nextA:=sourceNext A p base length start active a carried accumulated transition related.sourceColumns related.sourceMask ha hga hla
    let nextB:=modelNext A p base length start active b carried accumulated transition related.modelColumns related.modelMask hb hgb hlb
    have nextRelated : Related p base nextA.val nextB.val :=
      next_related A p base length start active a b carried accumulated transition related ha hb hga hgb hla hlb
    obtain ⟨x,y,hx,hy,hr⟩:=loop_related A p base length (start+1) (by omega) nextA.val nextB.val nextRelated
    refine ⟨x,y,?_,?_,hr⟩
    · rw [nextA.property]
      simpa only [Option.bind_some,Nat.cast_add,Nat.cast_one] using hx
    · rw [nextB.property]
      simpa only [Option.bind_some] using hy
  · have done : length≤start := by omega
    have doneInt : (length : Int)≤start := by exact_mod_cast done
    rw [stepForRangeIntAux.step_ge (by decide) doneInt,stepForRangeAux.step_ge (by decide) done]
    exact ⟨a,b,rfl,rfl,related⟩
termination_by length-start
decreasing_by omega




#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.FlatBackwardTransport
