import bench.tritonbench_g.l2_norm_bwd.Proofs.Scalar
import bench.tritonbench_g.swiglu_backward.Proofs.Mathematical_WholeTile

namespace VeriTile.Bench.Repairs.L2NormBwd.Tile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole (writeCell scatter_mem fold_written fold_frame)
noncomputable section
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000

def rowState {α : Type} [Inhabited α] (row : Nat) (eps : α) (s : State α) : State α :=
  {s with pids:=fun ax => if ax=0 then row else 0
          regs:=(State.setReg s (dtype := .real) (shape := []) "eps" (fun _ => eps)).regs}

private theorem index_toNat (row stride lane col : Nat) :
    ((row : Int)*(stride : Int)+(lane : Int)*(col : Int)).toNat=row*stride+lane*col := by
  rw [←Int.natCast_mul,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

private theorem index_lt (lane N : Nat) : ((lane : Int)<(N : Int)) ↔ lane<N := by omega

def input {α : Type} [Inhabited α] (s : State α) (region : RegionName) (row stride col : Nat) : Nat → α :=
  fun i => (s.mem region (row*stride+i*col)).read .real

def tileScalar {α : Type} (M : Algebra α) (N B : Nat) (eps : α) (xs dys : Nat → α) (i : Fin B) : α :=
  let x := masked M N B xs i
  let dy := masked M N B dys i
  let q := M.binary (some .fp32) .real .add (productSum M N B xs xs) eps
  let rstd := M.binary (some .fp32) .real .div (M.literal (some .fp32) .real 1)
    (M.unary (some .fp32) .sqrt q)
  M.binary (some .fp32) .real .sub
    (M.binary (some .fp32) .real .mul dy rstd)
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .mul (productSum M N B dys xs)
          (M.binary (some .fp32) .real .div (M.literal (some .fp32) .real 1) q)) rstd) x)

theorem tileScalar_active {α : Type} (M : Algebra α) (N B : Nat) (eps : α)
    (xs dys : Nat → α) (i : Fin B) (active : i.val<N) :
    tileScalar M N B eps xs dys i=scalar M N B eps xs dys i.val := by
  simp only [tileScalar,scalar,masked,loaded,active,if_true]

structure Invocation (α : Type) where
  alg : Algebra α
  layout : Strides
  N : Nat
  B : Nat
  row : Nat
  eps : α
  initial : State α

def Invocation.xs {α : Type} [Inhabited α] (c : Invocation α) := input c.initial "X" c.row c.layout.xRow c.layout.xCol
def Invocation.dys {α : Type} [Inhabited α] (c : Invocation α) := input c.initial "DY" c.row c.layout.dyRow c.layout.dyCol
def Invocation.q {α : Type} [Inhabited α] (c : Invocation α) := c.alg.binary (some .fp32) .real .add (productSum c.alg c.N c.B c.xs c.xs) c.eps
def Invocation.rs {α : Type} [Inhabited α] (c : Invocation α) := c.alg.binary (some .fp32) .real .div (c.alg.literal (some .fp32) .real 1) (c.alg.unary (some .fp32) .sqrt c.q)
def statement {α : Type} (c : Invocation α) (k : Nat) := (source c.layout c.N c.B).surfaceBody.getD k (.effectMarker "out-of-range")

def state0 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  rowState c.row c.eps c.initial

def state1 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state0 c).setReg "row" .int [] (fun _ => (c.row : Int))

def state2 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state1 c).setReg "X" .ptr [] (fun _ => ("X",c.row*c.layout.xRow))

def state3 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state2 c).setReg "DX" .ptr [] (fun _ => ("DX",c.row*c.layout.dxRow))

def state4 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state3 c).setReg "DY" .ptr [] (fun _ => ("DY",c.row*c.layout.dyRow))

def state5 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state4 c).setReg "cols" .int [c.B] (fun i => (i.1.val : Int))

def state6 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state5 c).setReg "x" .real [c.B] (fun i => loaded c.alg c.N c.B c.xs i.1)

def state7 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state6 c).setReg "x" .real [c.B] (fun i => masked c.alg c.N c.B c.xs i.1)

def state8 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state7 c).setReg "var" .real [] (fun _ => productSum c.alg c.N c.B c.xs c.xs)

def state9 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state8 c).setReg "rstd" .real [] (fun _ => c.rs)

def state10 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state9 c).setReg "mask" .bool [c.B] (fun i => decide (i.1.val<c.N))

def state11 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state10 c).setReg "dy" .real [c.B] (fun i => loaded c.alg c.N c.B c.dys i.1)

def state12 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state11 c).setReg "dy" .real [c.B] (fun i => masked c.alg c.N c.B c.dys i.1)

def state13 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (state12 c).setReg "dx" .real [c.B] (fun i => tileScalar c.alg c.N c.B c.eps c.xs c.dys i.1)

def state14 {α : Type} [Inhabited α] (c : Invocation α) : State α :=
  (TileShape.allIndices [c.B]).foldl (fun t i => if i.1.val<c.N then
  t.write "DX" (c.row*c.layout.dxRow+i.1.val*c.layout.dxCol) (.mk .real (tileScalar c.alg c.N c.B c.eps c.xs c.dys i.1)) else t) (state13 c)

theorem statement0 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 0) (state0 c)=some (state1 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try simp only [show (0.0:ℝ)=0 from by norm_num, id_eq]
  try rfl

#axiomsClean statement0

theorem statement1 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 1) (state1 c)=some (state2 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2]
  congr 2
  funext i
  change (("X":RegionName),0+((c.row:Int)*(c.layout.xRow:Int)).toNat)=(("X":RegionName),c.row*c.layout.xRow)
  simp only [←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]

#axiomsClean statement1

theorem statement2 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 2) (state2 c)=some (state3 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3]
  congr 2
  funext i
  change (("DX":RegionName),0+((c.row:Int)*(c.layout.dxRow:Int)).toNat)=(("DX":RegionName),c.row*c.layout.dxRow)
  simp only [←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]

#axiomsClean statement2

theorem statement3 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 3) (state3 c)=some (state4 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3,state4]
  congr 2
  funext i
  change (("DY":RegionName),0+((c.row:Int)*(c.layout.dyRow:Int)).toNat)=(("DY":RegionName),c.row*c.layout.dyRow)
  simp only [←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]

#axiomsClean statement3

theorem statement4 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 4) (state4 c)=some (state5 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3,state4,state5]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try simp only [show (0.0:ℝ)=0 from by norm_num, id_eq]
  try rfl

#axiomsClean statement4

theorem statement5 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 5) (state5 c)=some (state6 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3,state4,state5,state6]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try simp only [show (0.0:ℝ)=0 from by norm_num, id_eq]
  try rfl
  all_goals try congr 2
  all_goals try funext i
  all_goals simp [Int.toNat_mul,Value,FloatDType.toTileDType,bop,Broadcast.leftIndex,Broadcast.rightIndex]

#axiomsClean statement5

theorem statement6 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 6) (state6 c)=some (state7 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3,state4,state5,state6,state7]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try simp only [show (0.0:ℝ)=0 from by norm_num, id_eq]
  try rfl
  all_goals try congr 2
  all_goals try funext i
  all_goals simp [Int.toNat_mul,Value,FloatDType.toTileDType,bop,Broadcast.leftIndex,Broadcast.rightIndex]

#axiomsClean statement6

theorem statement7 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 7) (state7 c)=some (state8 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3,state4,state5,state6,state7,state8]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try simp only [show (0.0:ℝ)=0 from by norm_num, id_eq]
  try rfl

#axiomsClean statement7

theorem statement8 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 8) (state8 c)=some (state9 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3,state4,state5,state6,state7,state8,state9]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try simp only [show (0.0:ℝ)=0 from by norm_num, id_eq]
  try rfl

#axiomsClean statement8

theorem statement9 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 9) (state9 c)=some (state10 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3,state4,state5,state6,state7,state8,state9,state10]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try simp only [show (0.0:ℝ)=0 from by norm_num, id_eq]
  try rfl
  all_goals try congr 2
  all_goals try funext i
  all_goals simp [Int.toNat_mul,Value,FloatDType.toTileDType,bop,Broadcast.leftIndex,Broadcast.rightIndex]

#axiomsClean statement9

theorem statement10 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 10) (state10 c)=some (state11 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3,state4,state5,state6,state7,state8,state9,state10,state11]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try simp only [show (0.0:ℝ)=0 from by norm_num, id_eq]
  try rfl
  all_goals try congr 2
  all_goals try funext i
  all_goals simp [Int.toNat_mul,Value,FloatDType.toTileDType,bop,Broadcast.leftIndex,Broadcast.rightIndex]

#axiomsClean statement10

theorem statement11 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 11) (state11 c)=some (state12 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3,state4,state5,state6,state7,state8,state9,state10,state11,state12]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try simp only [show (0.0:ℝ)=0 from by norm_num, id_eq]
  try rfl
  all_goals try congr 2
  all_goals try funext i
  all_goals simp [Int.toNat_mul,Value,FloatDType.toTileDType,bop,Broadcast.leftIndex,Broadcast.rightIndex]

#axiomsClean statement11

theorem statement12 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 12) (state12 c)=some (state13 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3,state4,state5,state6,state7,state8,state9,state10,state11,state12,state13]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try simp only [show (0.0:ℝ)=0 from by norm_num, id_eq]
  try rfl

#axiomsClean statement12

theorem statement13 {α : Type} [Inhabited α] (c : Invocation α) :
    step c.alg (statement c 13) (state13 c)=some (state14 c) := by
  simp only [statement,source,kernel,ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,Option.bind_eq_bind,Option.pure_def,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.bop,FP.Structural.natLt,rowState,FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,Region.cast,index_toNat,index_lt,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,decide_eq_true_eq,ite_true,ite_false,Nat.zero_add,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,loaded,masked,productSum,tileScalar,input,Invocation.xs,Invocation.dys,Invocation.q,Invocation.rs,state0,state1,state2,state3,state4,state5,state6,state7,state8,state9,state10,state11,state12,state13,state14]
  try simp only [Option.bind_eq_bind,Option.pure_def,Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  try simp only [show (0.0:ℝ)=0 from by norm_num, id_eq]
  try rfl

#axiomsClean statement13

/-- Construct every statement, with cached individual operational proofs. -/
def sourceRun {α : Type} [Inhabited α] (M : Algebra α) (layout : Strides) (N B row : Nat)
    (eps : α) (s : State α) :
    {t : State α // run M (source layout N B).surfaceBody (rowState row eps s)=some t} := by
  let c : Invocation α := ⟨M,layout,N,B,row,eps,s⟩
  refine ⟨state14 c,?_⟩
  change run c.alg [statement c 0,statement c 1,statement c 2,statement c 3,statement c 4,statement c 5,statement c 6,statement c 7,statement c 8,statement c 9,statement c 10,statement c 11,statement c 12,statement c 13] (state0 c)=some (state14 c)
  simp only [run,statement0,statement1,statement2,statement3,statement4,statement5,statement6,statement7,statement8,statement9,statement10,statement11,statement12,statement13,Option.bind,Option.bind_eq_bind,Option.pure_def]

#axiomsClean sourceRun
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Tile
