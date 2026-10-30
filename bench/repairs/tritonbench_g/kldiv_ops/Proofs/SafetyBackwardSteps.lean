import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Backward
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.KldivOps.SafetyBackward
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def command (p : Parameters) (flag : Bool) (i : Nat) : ComputeStmt :=
  (Backward.body p flag).getD i (.effectMarker "")
def b0 (s : State α) (start : Nat) := s.setReg "i" .int [] (fun _ => (start:Int))
def b1 (p : Parameters) (s : State α) (start : Nat) :=
  (b0 s start).setReg "offsets" .int [p.block] (fun j => (start:Int)+(j.1.val:Int))
def b2 (p : Parameters) (s : State α) (start : Nat) :=
  (b1 p s start).setReg "mask" .bool [p.block] (fun j => decide (start+j.1.val<p.cols))
def raw (M : Algebra α) (p : Parameters) (s : State α) (row start : Nat) (j : TileIndex [p.block]) : α :=
  if start+j.1.val<p.cols then (s.mem p.T (row*p.cols+(start+j.1.val))).read .real
  else M.literal (some .fp32) .real 0
def b3 (M : Algebra α) (p : Parameters) (s : State α) (row start : Nat) :=
  (b2 p s start).setReg "target" .real [p.block] (fun j => M.fp32Load (raw M p s row start j))
def b4 (M : Algebra α) (p : Parameters) (flag : Bool) (s : State α) (row start : Nat) :=
  (b3 M p s row start).setReg "res" .real [p.block] (fun j => Backward.scalar M flag (raw M p s row start j))

theorem step0 (M : Algebra α) (p : Parameters) (flag : Bool) (s : State α) (start : Nat) :
    step (engine M) (command p flag 0) (b0 s start)=some (b1 p s start) := by
  cases flag
  all_goals simp only [command,Backward.body,backward,KernelDSL.backward,KernelDSL.backwardDefault,KernelDSL.backwardLog,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLog,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefault,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefaultOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLogOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.commuteOutput,
    ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,b0,b1,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals norm_num only
  all_goals try simp only [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.ofNat_lt]
  all_goals rfl

theorem step1 (M : Algebra α) (p : Parameters) (flag : Bool) (s : State α) (start : Nat) :
    step (engine M) (command p flag 1) (b1 p s start)=some (b2 p s start) := by
  cases flag
  all_goals simp only [command,Backward.body,backward,KernelDSL.backward,KernelDSL.backwardDefault,KernelDSL.backwardLog,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLog,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefault,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefaultOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLogOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.commuteOutput,
    ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,b0,b1,b2,b3,b4,raw,Backward.scalar,ofFloat,toFloat,
    ←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,Tile.active_int,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals norm_num only
  all_goals try simp only [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.ofNat_lt]
  all_goals
    change some ((b1 p s start).setReg "mask" .bool [p.block]
      (FP.Structural.bop (fun a b : Int => decide (a<b)) .scalarR
        (fun j => ((start+j.1.val:Nat):Int)) (fun _ => (p.cols:Int)))) =
      some ((b1 p s start).setReg "mask" .bool [p.block]
        (fun j => decide (start+j.1.val<p.cols)))
    congr 2
    funext j
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Int.ofNat_lt]

theorem step2 (M : Algebra α) (p : Parameters) (flag : Bool) (s : State α) (row start : Nat)
    (h : Backward.Registers p row s) :
    step (engine M) (command p flag 2) (b2 p s start)=some (b3 M p s row start) := by
  cases flag
  all_goals simp only [command,Backward.body,backward,KernelDSL.backward,KernelDSL.backwardDefault,KernelDSL.backwardLog,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLog,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefault,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefaultOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLogOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.commuteOutput,
    ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,b0,b1,b2,b3,b4,raw,Backward.scalar,ofFloat,toFloat,
    ←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,Tile.active_int,h.target,h.output,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals norm_num only
  all_goals try simp only [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.ofNat_lt]
  all_goals rfl

theorem step3 (M : Algebra α) (p : Parameters) (flag : Bool) (s : State α) (row start : Nat) :
    step (engine M) (command p flag 3) (b3 M p s row start)=some (b4 M p flag s row start) := by
  cases flag
  all_goals simp only [command,Backward.body,backward,KernelDSL.backward,KernelDSL.backwardDefault,KernelDSL.backwardLog,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLog,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefault,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefaultOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLogOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.commuteOutput,
    ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,b0,b1,b2,b3,b4,raw,Backward.scalar,ofFloat,toFloat,
    ←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,Tile.active_int,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals norm_num only
  all_goals try simp only [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.ofNat_lt]
  all_goals rfl

end
end VeriTile.Bench.Repairs.KldivOps.SafetyBackward
