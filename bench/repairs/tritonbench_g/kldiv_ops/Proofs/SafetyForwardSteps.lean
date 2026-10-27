import bench.repairs.tritonbench_g.kldiv_ops.Proofs.SafetyBackwardSteps
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Forward

namespace VeriTile.Bench.Repairs.KldivOps.SafetyForward
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

abbrev f0 := @SafetyBackward.b0
abbrev f1 := @SafetyBackward.b1
abbrev f2 := @SafetyBackward.b2

def command (p : Parameters) (flag : Bool) (red i : Nat) : ComputeStmt :=
  (forwardBody p flag red).getD i (.effectMarker "")
def raw (M : Algebra α) (p : Parameters) (s : State α) (r : RegionName)
    (row start : Nat) (j : TileIndex [p.block]) : α :=
  if start+j.1.val<p.cols then (s.mem r (row*p.cols+(start+j.1.val))).read .real
  else Tile.zero M

def f3 (M : Algebra α) (p : Parameters) (s : State α) (row start : Nat) :=
  (f2 p s start).setReg "y" .real [p.block] (fun j => M.fp32Load (raw M p s p.Y row start j))
def f4 (M : Algebra α) (p : Parameters) (s : State α) (row start : Nat) :=
  (f3 M p s row start).setReg "y_true" .real [p.block] (fun j => M.fp32Load (raw M p s p.T row start j))
def f5 (M : Algebra α) (p : Parameters) (flag : Bool) (eps : α)
    (s : State α) (row start : Nat) :=
  (f4 M p s row start).setReg "loss" .real [p.block]
    (fun j => Tile.scalar M flag eps (raw M p s p.Y row start j) (raw M p s p.T row start j))

theorem step0 (M : Algebra α) (p : Parameters) (flag : Bool) (red row start : Nat)
    (eps carry : α) (s : State α) (h : Tile.Registers p red row eps carry s) :
    step (engine M) (command p flag red 0) (f0 s start)=some (f1 p s start) := by
  cases flag
  all_goals simp only [command,forwardBody,forwardTemplateBody,forwardTemplate,KernelDSL.forward,KernelDSL.forwardDefault,KernelDSL.forwardLog,KernelDSL.forwardBody,epsilonLoss,
    ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    List.take,List.drop,List.nil_append,List.cons_append,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,
    f0,f1,f2,SafetyBackward.b0,SafetyBackward.b1,SafetyBackward.b2,f3,f4,f5,raw,Tile.scalar,Tile.zero,ofFloat,toFloat,FloatDType.toTileDType,id_eq,Function.comp_apply,
    ←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,Tile.active_int,h.y,h.t,h.o,h.base,h.epsilon,h.sum,h.cols,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals norm_num only
  all_goals rfl

theorem step1 (M : Algebra α) (p : Parameters) (flag : Bool) (red row start : Nat)
    (eps carry : α) (s : State α) (h : Tile.Registers p red row eps carry s) :
    step (engine M) (command p flag red 1) (f1 p s start)=some (f2 p s start) := by
  cases flag
  all_goals simp only [command,forwardBody,forwardTemplateBody,forwardTemplate,KernelDSL.forward,KernelDSL.forwardDefault,KernelDSL.forwardLog,KernelDSL.forwardBody,epsilonLoss,
    ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    List.take,List.drop,List.nil_append,List.cons_append,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,
    f0,f1,f2,SafetyBackward.b0,SafetyBackward.b1,SafetyBackward.b2,f3,f4,f5,raw,Tile.scalar,Tile.zero,ofFloat,toFloat,FloatDType.toTileDType,id_eq,Function.comp_apply,
    ←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,Tile.active_int,h.y,h.t,h.o,h.base,h.epsilon,h.sum,h.cols,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals norm_num only
  all_goals
    change some ((f1 p s start).setReg "mask" .bool [p.block]
      (FP.Structural.bop (fun a b : Int => decide (a<b)) .scalarR
        (fun j => ((start+j.1.val:Nat):Int)) (fun _ => (p.cols:Int)))) =
      some ((f1 p s start).setReg "mask" .bool [p.block]
        (fun j => decide (start+j.1.val<p.cols)))
    congr 2
    funext j
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Int.ofNat_lt]

theorem step2 (M : Algebra α) (p : Parameters) (flag : Bool) (red row start : Nat)
    (eps carry : α) (s : State α) (h : Tile.Registers p red row eps carry s) :
    step (engine M) (command p flag red 2) (f2 p s start)=some (f3 M p s row start) := by
  cases flag
  all_goals simp only [command,forwardBody,forwardTemplateBody,forwardTemplate,KernelDSL.forward,KernelDSL.forwardDefault,KernelDSL.forwardLog,KernelDSL.forwardBody,epsilonLoss,
    ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    List.take,List.drop,List.nil_append,List.cons_append,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,
    f0,f1,f2,SafetyBackward.b0,SafetyBackward.b1,SafetyBackward.b2,f3,f4,f5,raw,Tile.scalar,Tile.zero,ofFloat,toFloat,FloatDType.toTileDType,id_eq,Function.comp_apply,
    ←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,Tile.active_int,h.y,h.t,h.o,h.base,h.epsilon,h.sum,h.cols,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals norm_num only
  all_goals rfl

theorem step3 (M : Algebra α) (p : Parameters) (flag : Bool) (red row start : Nat)
    (eps carry : α) (s : State α) (h : Tile.Registers p red row eps carry s) :
    step (engine M) (command p flag red 3) (f3 M p s row start)=some (f4 M p s row start) := by
  cases flag
  all_goals simp only [command,forwardBody,forwardTemplateBody,forwardTemplate,KernelDSL.forward,KernelDSL.forwardDefault,KernelDSL.forwardLog,KernelDSL.forwardBody,epsilonLoss,
    ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    List.take,List.drop,List.nil_append,List.cons_append,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,
    f0,f1,f2,SafetyBackward.b0,SafetyBackward.b1,SafetyBackward.b2,f3,f4,f5,raw,Tile.scalar,Tile.zero,ofFloat,toFloat,FloatDType.toTileDType,id_eq,Function.comp_apply,
    ←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,Tile.active_int,h.y,h.t,h.o,h.base,h.epsilon,h.sum,h.cols,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals norm_num only
  all_goals rfl

theorem step4 (M : Algebra α) (p : Parameters) (flag : Bool) (red row start : Nat)
    (eps carry : α) (s : State α) (h : Tile.Registers p red row eps carry s) :
    step (engine M) (command p flag red 4) (f4 M p s row start)=some (f5 M p flag eps s row start) := by
  cases flag
  all_goals simp only [command,forwardBody,forwardTemplateBody,forwardTemplate,KernelDSL.forward,KernelDSL.forwardDefault,KernelDSL.forwardLog,KernelDSL.forwardBody,epsilonLoss,
    ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    List.take,List.drop,List.nil_append,List.cons_append,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,
    f0,f1,f2,SafetyBackward.b0,SafetyBackward.b1,SafetyBackward.b2,f3,f4,f5,raw,Tile.scalar,Tile.zero,ofFloat,toFloat,FloatDType.toTileDType,id_eq,Function.comp_apply,
    ←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,Tile.active_int,h.y,h.t,h.o,h.base,h.epsilon,h.sum,h.cols,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals norm_num only
  all_goals rfl

end
end VeriTile.Bench.Repairs.KldivOps.SafetyForward
