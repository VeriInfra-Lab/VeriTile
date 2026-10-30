import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Tile

namespace VeriTile.Bench.Repairs.KldivTriton.Tile
open VeriTile.Triton FP.Structural
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

structure Invocation (α : Type) where
  m : Model α
  sy : Nat
  st : Nat
  so : Nat
  N : Nat
  B : Nat
  flag : Bool
  reduction : Nat
  yBase : Nat
  tBase : Nat
  oBase : Nat
  current : Nat
  initial : State α

def command {α : Type} (c : Invocation α) (k : Nat) : ComputeStmt :=
  (forwardSource c.sy c.st c.so c.N c.B c.flag c.reduction).surfaceBody.getD k (.effectMarker "out of range")

def bodyCommand {α : Type} (c : Invocation α) (k : Nat) : ComputeStmt :=
  (forwardBody c.sy c.st c.so c.N c.B c.flag c.reduction).getD k (.effectMarker "out of range")

def a0 {α : Type} (c : Invocation α) := c.initial
def a1 {α : Type} (c : Invocation α) := (a0 c).setReg "pid" .int [] (fun _ => (c.initial.pids 0 : Int))
def a2 {α : Type} (c : Invocation α) := (a1 c).setReg "y_ptr" .ptr [] (fun _ => ("y_ptr", c.initial.pids 0 * c.sy))
def a3 {α : Type} (c : Invocation α) := (a2 c).setReg "gt_ptr" .ptr [] (fun _ => ("gt_ptr", c.initial.pids 0 * c.st))
def a4 {α : Type} (c : Invocation α) := (a3 c).setReg "loss_ptr" .ptr [] (fun _ => ("loss_ptr", c.initial.pids 0 * c.so))
def a5 {α : Type} (c : Invocation α) := (a4 c).setReg "base_offsets" .nat [c.B] (fun j => j.1.val)

structure Ready {α : Type} (c : Invocation α) : Prop where
  i : c.initial.regs .int [] "i" = some (fun _ => (c.current : Int))
  y : c.initial.regs .ptr [] "y_ptr" = some (fun _ => (("y_ptr" : RegionName), c.yBase))
  t : c.initial.regs .ptr [] "gt_ptr" = some (fun _ => (("gt_ptr" : RegionName), c.tBase))
  o : c.initial.regs .ptr [] "loss_ptr" = some (fun _ => (("loss_ptr" : RegionName), c.oBase))
  base : c.initial.regs .nat [c.B] "base_offsets" = some (fun j => j.1.val)

def b0 {α : Type} (c : Invocation α) := c.initial
def b1 {α : Type} (c : Invocation α) := f1 c.initial c.B c.current
def b2 {α : Type} (c : Invocation α) := f2 c.initial c.N c.B c.current
def b3 {α : Type} [Inhabited α] (c : Invocation α) := f3 c.m c.initial c.yBase c.N c.B c.current
def b4 {α : Type} [Inhabited α] (c : Invocation α) := f4 c.m c.initial c.yBase c.tBase c.N c.B c.current
def b5 {α : Type} [Inhabited α] (c : Invocation α) := f5 c.m c.initial c.yBase c.tBase c.N c.B c.current c.flag
def b6 {α : Type} [Inhabited α] (c : Invocation α) := f6 c.m c.initial c.yBase c.tBase c.N c.B c.current c.flag
def b7 {α : Type} [Inhabited α] (c : Invocation α) := forwardResult c.m c.initial c.yBase c.tBase c.oBase c.N c.B c.current c.flag c.reduction

private theorem index_lt (a b N : Nat) : ((a : Int) + (b : Int) < (N : Int)) ↔ a + b < N := by omega

theorem initialize0 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.m.alg (command c 0) (a0 c) = some (a1 c) := by
  simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,a0,a1]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try rfl
  try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    change ((_:RegionName),0+((c.initial.pids 0:Int)*(c.sy:Int)).toNat)=_
    simp only [←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]

#axiomsClean initialize0

theorem initialize1 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.m.alg (command c 1) (a1 c) = some (a2 c) := by
  simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,a0,a1,a2]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try rfl
  try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    change ((_:RegionName),0+((c.initial.pids 0:Int)*(c.sy:Int)).toNat)=_
    simp only [←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]

#axiomsClean initialize1

theorem initialize2 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.m.alg (command c 2) (a2 c) = some (a3 c) := by
  simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,a0,a1,a2,a3]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try rfl
  try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    change ((_:RegionName),0+((c.initial.pids 0:Int)*(c.st:Int)).toNat)=_
    simp only [←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]

#axiomsClean initialize2

theorem initialize3 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.m.alg (command c 3) (a3 c) = some (a4 c) := by
  simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,a0,a1,a2,a3,a4]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try rfl
  try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    change ((_:RegionName),0+((c.initial.pids 0:Int)*(c.so:Int)).toNat)=_
    simp only [←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]

#axiomsClean initialize3

theorem initialize4 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.m.alg (command c 4) (a4 c) = some (a5 c) := by
  simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,a0,a1,a2,a3,a4,a5]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try rfl
  try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    change ((_:RegionName),0+((c.initial.pids 0:Int)*(c.so:Int)).toNat)=_
    simp only [←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]

#axiomsClean initialize4

theorem block0 {α : Type} [Inhabited α] (c : Invocation α) (h : Ready c) :
    step c.m.alg (bodyCommand c 0) (b0 c) = some (b1 c) := by
  all_goals simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,b0,b1,f1,zero,loaded,input,laneLoss,loss,rawLoss,forwardResult,scatter,h.i,h.y,h.t,h.o,h.base,c.m.eq_ok]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add]
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
  all_goals simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,b0,b1,b2,f1,f2,zero,loaded,input,laneLoss,loss,rawLoss,forwardResult,scatter,h.i,h.y,h.t,h.o,h.base,c.m.eq_ok]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add]
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
  all_goals simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,b0,b1,b2,b3,f1,f2,f3,zero,loaded,input,laneLoss,loss,rawLoss,forwardResult,scatter,h.i,h.y,h.t,h.o,h.base,c.m.eq_ok]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add]
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
  all_goals simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,b0,b1,b2,b3,b4,f1,f2,f3,f4,zero,loaded,input,laneLoss,loss,rawLoss,forwardResult,scatter,h.i,h.y,h.t,h.o,h.base,c.m.eq_ok]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add]
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
  all_goals simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,b0,b1,b2,b3,b4,b5,f1,f2,f3,f4,f5,zero,loaded,input,laneLoss,loss,rawLoss,forwardResult,scatter,h.i,h.y,h.t,h.o,h.base,c.m.eq_ok,hf,run,List.nil_append,List.cons_append]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega

#axiomsClean block4

theorem block5 {α : Type} [Inhabited α] (c : Invocation α) (h : Ready c) :
    step c.m.alg (bodyCommand c 5) (b5 c) = some (b6 c) := by
  cases hf : c.flag
  all_goals simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,b0,b1,b2,b3,b4,b5,b6,f1,f2,f3,f4,f5,f6,zero,loaded,input,laneLoss,loss,rawLoss,forwardResult,scatter,h.i,h.y,h.t,h.o,h.base,c.m.eq_ok,hf]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega

#axiomsClean block5

theorem block6 {α : Type} [Inhabited α] (c : Invocation α) (h : Ready c) :
    step c.m.alg (bodyCommand c 6) (b6 c) = some (b7 c) := by
  by_cases hr : c.reduction = 0 <;> cases hf : c.flag
  all_goals simp only [command,bodyCommand,forwardSource,forwardTemplate,forwardBody,ComputeKernel.surfaceBody,
      List.take,List.drop,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
      Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_lt,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
      decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.allIndices,List.finRange_succ,List.foldl_cons,List.foldl_nil,List.flatMap_cons,List.flatMap_nil,
      State.write_regs,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,b0,b1,b2,b3,b4,b5,b6,b7,f1,f2,f3,f4,f5,f6,zero,loaded,input,laneLoss,loss,rawLoss,forwardResult,scatter,h.i,h.y,h.t,h.o,h.base,c.m.eq_ok,hr,hf]
  all_goals try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Nat.zero_add]
  all_goals try simp only [show (0.0 : ℝ) = 0 from by norm_num,id_eq]
  all_goals try rfl
  all_goals try
    congr 2
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    first | rfl | (simp only [Int.ofNat_lt]) | omega

#axiomsClean block6

theorem forward_initialize {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat) :
    run m.alg ((forwardSource sy st so N B flag reduction).surfaceBody.take 5) s =
      some (forwardInitial s sy st so B) := by
  let c : Invocation α := ⟨m,sy,st,so,N,B,flag,reduction,0,0,0,0,s⟩
  change run c.m.alg [command c 0,command c 1,command c 2,command c 3,command c 4]
    (a0 c) = some (a5 c)
  simp only [run,initialize0,initialize1,initialize2,initialize3,initialize4,
    Option.bind,Option.bind_eq_bind,Option.pure_def]

theorem forward_step {α : Type} [Inhabited α] (c : Invocation α) (h : Ready c) :
    run c.m.alg (forwardBody c.sy c.st c.so c.N c.B c.flag c.reduction) c.initial =
      some (b7 c) := by
  change run c.m.alg [bodyCommand c 0,bodyCommand c 1,bodyCommand c 2,bodyCommand c 3,
    bodyCommand c 4,bodyCommand c 5,bodyCommand c 6] (b0 c) = some (b7 c)
  simp only [run,block0 c h,block1 c h,block2 c h,block3 c h,block4 c h,block5 c h,block6 c h,
    Option.bind,Option.bind_eq_bind,Option.pure_def]

#axiomsClean forward_initialize
#axiomsClean forward_step
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Tile
