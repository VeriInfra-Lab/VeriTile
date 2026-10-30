import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Statement

namespace VeriTile.Bench.Repairs.KldivTriton.Backward
open VeriTile.Triton FP.Structural
open Tile (Model Invocation loaded input gradient scatter)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

def body (so st N B : Nat) (flag : Bool) : List ComputeStmt :=
  match ((backwardSource so st N B flag).surfaceBody[5]? : Option ComputeStmt) with
  | some (.forRangeInt _ _ _ _ code) => code
  | _ => []
def command {α : Type} (c : Invocation α) (k : Nat) : ComputeStmt :=
  (backwardSource c.so c.st c.N c.B c.flag).surfaceBody.getD k (.effectMarker "out of range")
def bodyCommand {α : Type} (c : Invocation α) (k : Nat) : ComputeStmt :=
  (body c.so c.st c.N c.B c.flag).getD k (.effectMarker "out of range")

def a0 {α : Type} (c : Invocation α) := c.initial
def a1 {α : Type} (c : Invocation α) := (a0 c).setReg "pid" .int [] (fun _ => (c.initial.pids 0 : Int))
def a2 {α : Type} (c : Invocation α) := (a1 c).setReg "input_ptr" .ptr [] (fun _ => ("input_ptr", c.initial.pids 0 * c.so))
def a3 {α : Type} (c : Invocation α) := (a2 c).setReg "target_ptr" .ptr [] (fun _ => ("target_ptr", c.initial.pids 0 * c.st))
def a4 {α : Type} (c : Invocation α) := (a3 c).setReg "offsets" .int [c.B] (fun j => (j.1.val : Int))
def a5 {α : Type} (c : Invocation α) := (a4 c).setReg "mask" .bool [c.B] (fun j => decide (j.1.val < c.N))

structure Ready {α : Type} (c : Invocation α) : Prop where
  i : c.initial.regs .int [] "i" = some (fun _ => (c.current : Int))
  t : c.initial.regs .ptr [] "target_ptr" = some (fun _ => (("target_ptr" : RegionName), c.tBase))
  o : c.initial.regs .ptr [] "input_ptr" = some (fun _ => (("input_ptr" : RegionName), c.oBase))

def b0 {α : Type} (c : Invocation α) := c.initial
def b1 {α : Type} (c : Invocation α) := c.initial.setReg "offsets" .int [c.B] (fun j => ((c.current + j.1.val : Nat) : Int))
def b2 {α : Type} (c : Invocation α) := (b1 c).setReg "mask" .bool [c.B] (fun j => decide (c.current + j.1.val < c.N))
def b3 {α : Type} [Inhabited α] (c : Invocation α) := (b2 c).setReg "target" .real [c.B]
  (fun j => loaded c.m c.N (c.current + j.1.val) (input c.initial "target_ptr" c.tBase))
def b4 {α : Type} [Inhabited α] (c : Invocation α) := (b3 c).setReg "res" .real [c.B]
  (fun j => gradient c.m c.flag (loaded c.m c.N (c.current + j.1.val) (input c.initial "target_ptr" c.tBase)))
def b5 {α : Type} [Inhabited α] (c : Invocation α) := scatter (b4 c) "input_ptr" c.oBase c.N c.B c.current
  (fun j => gradient c.m c.flag (loaded c.m c.N (c.current + j.1.val) (input c.initial "target_ptr" c.tBase)))

theorem initialize0 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.m.alg (command c 0) (a0 c) = some (a1 c) := by
  all_goals simp only [command,bodyCommand,body,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    a0,a1,loaded,input,gradient,Tile.zero,scatter]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add,Int.ofNat_lt]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega

#axiomsClean initialize0

theorem initialize1 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.m.alg (command c 1) (a1 c) = some (a2 c) := by
  all_goals simp only [command,bodyCommand,body,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    a0,a1,a2,loaded,input,gradient,Tile.zero,scatter]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add,Int.ofNat_lt]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega
  all_goals try
    congr 2
    funext i
    change ((_:RegionName),0+((c.initial.pids 0:Int)*(c.so:Int)).toNat)=_
    simp only [←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]

#axiomsClean initialize1

theorem initialize2 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.m.alg (command c 2) (a2 c) = some (a3 c) := by
  all_goals simp only [command,bodyCommand,body,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    a0,a1,a2,a3,loaded,input,gradient,Tile.zero,scatter]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add,Int.ofNat_lt]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega
  all_goals try
    congr 2
    funext i
    change ((_:RegionName),0+((c.initial.pids 0:Int)*(c.st:Int)).toNat)=_
    simp only [←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]

#axiomsClean initialize2

theorem initialize3 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.m.alg (command c 3) (a3 c) = some (a4 c) := by
  all_goals simp only [command,bodyCommand,body,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    a0,a1,a2,a3,a4,loaded,input,gradient,Tile.zero,scatter]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add,Int.ofNat_lt]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega

#axiomsClean initialize3

theorem initialize4 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.m.alg (command c 4) (a4 c) = some (a5 c) := by
  all_goals simp only [command,bodyCommand,body,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    a0,a1,a2,a3,a4,a5,loaded,input,gradient,Tile.zero,scatter]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add,Int.ofNat_lt]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega

#axiomsClean initialize4

theorem block0 {α : Type} [Inhabited α] (c : Invocation α) (h : Ready c) :
    step c.m.alg (bodyCommand c 0) (b0 c) = some (b1 c) := by
  all_goals simp only [command,bodyCommand,body,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    b0,b1,loaded,input,gradient,Tile.zero,scatter,h.i,h.t,h.o]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add,Int.ofNat_lt]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega

#axiomsClean block0

theorem block1 {α : Type} [Inhabited α] (c : Invocation α) (h : Ready c) :
    step c.m.alg (bodyCommand c 1) (b1 c) = some (b2 c) := by
  all_goals simp only [command,bodyCommand,body,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    b0,b1,b2,loaded,input,gradient,Tile.zero,scatter,h.i,h.t,h.o]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add,Int.ofNat_lt]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega

#axiomsClean block1

theorem block2 {α : Type} [Inhabited α] (c : Invocation α) (h : Ready c) :
    step c.m.alg (bodyCommand c 2) (b2 c) = some (b3 c) := by
  all_goals simp only [command,bodyCommand,body,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    b0,b1,b2,b3,loaded,input,gradient,Tile.zero,scatter,h.i,h.t,h.o]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add,Int.ofNat_lt]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega

#axiomsClean block2

theorem block3 {α : Type} [Inhabited α] (c : Invocation α) (h : Ready c) :
    step c.m.alg (bodyCommand c 3) (b3 c) = some (b4 c) := by
  cases hf : c.flag
  all_goals simp only [command,bodyCommand,body,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    b0,b1,b2,b3,b4,loaded,input,gradient,Tile.zero,scatter,h.i,h.t,h.o,hf]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add,Int.ofNat_lt]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega

#axiomsClean block3

theorem block4 {α : Type} [Inhabited α] (c : Invocation α) (h : Ready c) :
    step c.m.alg (bodyCommand c 4) (b4 c) = some (b5 c) := by
  cases hf : c.flag
  all_goals simp only [command,bodyCommand,body,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    b0,b1,b2,b3,b4,b5,loaded,input,gradient,Tile.zero,scatter,h.i,h.t,h.o,hf]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add,Int.ofNat_lt]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega

#axiomsClean block4

theorem source_initialize {α : Type} [Inhabited α] (c : Invocation α) :
    run c.m.alg ((backwardSource c.so c.st c.N c.B c.flag).surfaceBody.take 5) c.initial = some (a5 c) := by
  change run c.m.alg [command c 0,command c 1,command c 2,command c 3,command c 4] (a0 c) = some (a5 c)
  simp only [run,initialize0,initialize1,initialize2,initialize3,initialize4,Option.bind,Option.bind_eq_bind,Option.pure_def]

theorem block {α : Type} [Inhabited α] (c : Invocation α) (h : Ready c) :
    run c.m.alg (body c.so c.st c.N c.B c.flag) c.initial = some (b5 c) := by
  change run c.m.alg [bodyCommand c 0,bodyCommand c 1,bodyCommand c 2,bodyCommand c 3,bodyCommand c 4] (b0 c) = some (b5 c)
  simp only [run,block0 c h,block1 c h,block2 c h,block3 c h,block4 c h,Option.bind,Option.bind_eq_bind,Option.pure_def]

#axiomsClean source_initialize
#axiomsClean block
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Backward
