import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Source
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourceSafety
open VeriTile Triton FP.Structural Safety
open VeriTile.Bench.TritonBenchG.PowScalarTensor WholeTile
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def command (c : Parameters) (candidate : Bool) (i : Nat) : ComputeStmt :=
  (perTile c candidate).getD i (.effectMarker "")
def b1 (s : State α) (T : Nat) := s.setReg "tile_id0" .nat [] (fun _ => T)
def b2 (c : Parameters) (s : State α) (T : Nat) :=
  (b1 s T).setReg "offset0" .nat [] (fun _ => T*c.B)
def b3 (c : Parameters) (s : State α) (T : Nat) :=
  (b2 c s T).setReg "in0_bptr" .int [c.B]
    (fun j => c.input.base+((T*c.B+j.1.val:Nat):Int)*c.input.stride)
def b4 (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat) :=
  (b3 c s T).setReg "in0" .real [c.B] (inputValues M (config c candidate) T s)
def b5 (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat) :=
  (b4 M c candidate s T).setReg "out0" .real [c.B] (selectedValues M (config c candidate) T s)
def b6 (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat) :=
  (b5 M c candidate s T).setReg "out0_bptr" .int [c.B]
    (fun j => c.output.base+((T*c.B+j.1.val:Nat):Int)*c.output.stride)

theorem step0 (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat)
    (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) :
    step M (command c candidate 0) (s)=some (b1 s T) := by
  cases candidate
  all_goals simp only [command,perTile,oneTileKernel,oneTileCommuted,ComputeKernel.surfaceBody,
    List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,List.drop,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    ComputeDType.eraseDType,FloatDType.toTileDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,ofFloat,toFloat,
    b1,b2,b3,b4,b5,b6,config,inputValues,selectedValues,rawValues,scalarResult,scalarBase,
    SignedView.View.address,SignedView.View.signedAddress,ht,
    Nat.mul_comm,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,dite_true,dite_false,ite_true,ite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try rfl

theorem step1 (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat)
    (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) :
    step M (command c candidate 1) (b1 s T)=some (b2 c s T) := by
  cases candidate
  all_goals simp only [command,perTile,oneTileKernel,oneTileCommuted,ComputeKernel.surfaceBody,
    List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,List.drop,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    ComputeDType.eraseDType,FloatDType.toTileDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,ofFloat,toFloat,
    b1,b2,b3,b4,b5,b6,config,inputValues,selectedValues,rawValues,scalarResult,scalarBase,
    SignedView.View.address,SignedView.View.signedAddress,ht,
    Nat.mul_comm,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,dite_true,dite_false,ite_true,ite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try rfl
  change some ((b1 s T).setReg "offset0" .nat []
    (FP.Structural.bop (fun x y : Nat => x*y) .nil (fun _ => c.B) (fun _ => T))) =
      some ((b1 s T).setReg "offset0" .nat [] (fun _ => T*c.B))
  congr 2
  funext j
  exact Nat.mul_comm c.B T

theorem step2 (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat)
    (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) :
    step M (command c candidate 2) (b2 c s T)=some (b3 c s T) := by
  cases candidate
  all_goals simp only [command,perTile,oneTileKernel,oneTileCommuted,ComputeKernel.surfaceBody,
    List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,List.drop,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    ComputeDType.eraseDType,FloatDType.toTileDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,ofFloat,toFloat,
    b1,b2,b3,b4,b5,b6,config,inputValues,selectedValues,rawValues,scalarResult,scalarBase,
    SignedView.View.address,SignedView.View.signedAddress,ht,
    Nat.mul_comm,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,dite_true,dite_false,ite_true,ite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try rfl

theorem step3 (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat)
    (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) :
    step M (command c candidate 3) (b3 c s T)=some (b4 M c candidate s T) := by
  cases candidate
  all_goals simp only [command,perTile,oneTileKernel,oneTileCommuted,ComputeKernel.surfaceBody,
    List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,List.drop,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    ComputeDType.eraseDType,FloatDType.toTileDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,ofFloat,toFloat,
    b1,b2,b3,b4,b5,b6,config,inputValues,selectedValues,rawValues,scalarResult,scalarBase,
    SignedView.View.address,SignedView.View.signedAddress,ht,
    Nat.mul_comm,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,dite_true,dite_false,ite_true,ite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try rfl

theorem step4 (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat)
    (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) :
    step M (command c candidate 4) (b4 M c candidate s T)=some (b5 M c candidate s T) := by
  cases candidate
  all_goals simp only [command,perTile,oneTileKernel,oneTileCommuted,ComputeKernel.surfaceBody,
    List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,List.drop,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    ComputeDType.eraseDType,FloatDType.toTileDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,ofFloat,toFloat,
    b1,b2,b3,b4,b5,b6,config,inputValues,selectedValues,rawValues,scalarResult,scalarBase,
    SignedView.View.address,SignedView.View.signedAddress,ht,
    Nat.mul_comm,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,dite_true,dite_false,ite_true,ite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try rfl

theorem step5 (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat)
    (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) :
    step M (command c candidate 5) (b5 M c candidate s T)=some (b6 M c candidate s T) := by
  cases candidate
  all_goals simp only [command,perTile,oneTileKernel,oneTileCommuted,ComputeKernel.surfaceBody,
    List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,List.drop,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    ComputeDType.eraseDType,FloatDType.toTileDType,State.setReg,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,ofFloat,toFloat,
    b1,b2,b3,b4,b5,b6,config,inputValues,selectedValues,rawValues,scalarResult,scalarBase,
    SignedView.View.address,SignedView.View.signedAddress,ht,
    Nat.mul_comm,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,dite_true,dite_false,ite_true,ite_false,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try rfl

end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourceSafety
