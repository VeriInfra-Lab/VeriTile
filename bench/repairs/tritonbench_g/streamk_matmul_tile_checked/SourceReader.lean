import bench.repairs.tritonbench_g.streamk_matmul.Whole001.SourceDefinitions

/-! The task-local source subset keeps integer expressions, pointer updates,
masked loads, source loops and the FP16 conversion explicit. It is deliberately
not a compiler for unrelated Triton programs. Unsupported statement schemas
fail to decode. Extracted expressions drive execution. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
open Whole001

inductive Expr where
  | literal (n : Nat)
  | variable (name : String)
  | pid | laneM | laneN | laneK
  | signed64 (e : Expr)
  | add (a b : Expr) | sub (a b : Expr) | mul (a b : Expr)
  | div (a b : Expr) | rem (a b : Expr) | minimum (a b : Expr) | cdiv (a b : Expr)
  deriving DecidableEq, BEq, Repr
inductive Test where
  | eq (a b : Expr) | ne (a b : Expr) | lt (a b : Expr) | gt (a b : Expr)
  | and (a b : Test)
  | variable (name : String)
  deriving DecidableEq, BEq, Repr
inductive Pointer where
  | variable (name : String)
  | offset (p : Pointer) (e : Expr)
  deriving DecidableEq, BEq, Repr
inductive Statement where
  | integer (name : String) (e : Expr)
  | boolean (name : String) (e : Test)
  | pointer (name : String) (p : Pointer)
  | zeros (name : String) (m n : Expr)
  | load (name : String) (p : Pointer) (mask : Option Test)
  | dotAdd (acc left right : String)
  | half (name : String)
  | store (p : Pointer) (value : String) (mask : Option Test)
  | atomic (p : Pointer) (value : String) (mask : Test)
  | range (index : String) (lo hi : Expr) (body : List Statement)
  | branch (condition : Test) (yes no : List Statement)
  | loop (condition : Test) (body : List Statement)
  | tileCall (swizzle : Bool) (tile : Expr)
  | macCall
  | returnPair (a b : Expr)
  deriving Repr

abbrev Env := String → Nat
structure Lanes where
  pid : Nat := 0
  m : Nat := 0
  n : Nat := 0
  k : Nat := 0
  deriving Repr

def Expr.eval (env : Env) (lanes : Lanes) : Expr → Nat
  | .literal n => n
  | .variable name => env name
  | .pid => lanes.pid
  | .laneM => lanes.m
  | .laneN => lanes.n
  | .laneK => lanes.k
  | .signed64 e => e.eval env lanes
  | .add a b => a.eval env lanes+b.eval env lanes
  | .sub a b => a.eval env lanes-b.eval env lanes
  | .mul a b => a.eval env lanes*b.eval env lanes
  | .div a b => a.eval env lanes/b.eval env lanes
  | .rem a b => a.eval env lanes%b.eval env lanes
  | .minimum a b => min (a.eval env lanes) (b.eval env lanes)
  | .cdiv a b => (a.eval env lanes+b.eval env lanes-1)/b.eval env lanes

def Test.eval (env : Env) (lanes : Lanes) : Test → Prop
  | .eq a b => a.eval env lanes=b.eval env lanes
  | .ne a b => a.eval env lanes≠b.eval env lanes
  | .lt a b => a.eval env lanes<b.eval env lanes
  | .gt a b => b.eval env lanes<a.eval env lanes
  | .and a b => a.eval env lanes ∧ b.eval env lanes
  | .variable name => env name≠0
noncomputable instance (env : Env) (lanes : Lanes) (t : Test) : Decidable (t.eval env lanes) := by
  induction t <;> simp only [Test.eval] <;> infer_instance

def put (env : Env) (name : String) (value : Nat) : Env :=
  fun key => if key=name then value else env key

abbrev Bindings := List (String × Expr)
def bindings (xs : Bindings) (env : Env) (lanes : Lanes) : Env :=
  xs.foldl (fun current item => put current item.1 (item.2.eval current lanes)) env

/-- The integer straight-line subset is extracted, preserving assignment order. -/
def integers : List Statement → Option Bindings
  | [] => some []
  | .integer name e :: tail => return (name,e)::(← integers tail)
  | _ => none

structure Mapping where
  prepare : Bindings
  m : Expr
  n : Expr
  deriving Repr

def mapping : List Statement → Option Mapping
  | [.integer a x,.integer b y,.integer c z,.integer d w,.integer e v,
      .integer f r,.integer g s,.returnPair m n] =>
    some ⟨[(a,x),(b,y),(c,z),(d,w),(e,v),(f,r),(g,s)],m,n⟩
  | [.integer a x,.integer b y,.returnPair m n] => some ⟨[(a,x),(b,y)],m,n⟩
  | _ => none

structure Geometry where
  swizzle : Mapping
  linear : Mapping
  condition : Test
  tile : Expr
  deriving Repr

def geometry (swizzle linear : List Statement) (branch : Statement) : Option Geometry := do
  let s ← mapping swizzle
  let l ← mapping linear
  match branch with
  | .branch test [.tileCall Bool.true tile] [.tileCall Bool.false tile'] =>
    if tile=tile' then some ⟨s,l,test,tile⟩ else none
  | _ => none

structure DotLoop where
  index : String
  lo : Expr
  hi : Expr
  kk : Expr
  left : Pointer
  right : Pointer
  maskA : Test
  maskB : Test
  advanceA : Pointer
  advanceB : Pointer
  deriving Repr

def dotLoop : Statement → Option DotLoop
  | .range index lo hi
      [.integer "kk" kk,.load "a" left (some ma),.load "b" right (some mb),
       .dotAdd "acc" "a" "b",.pointer "A" stepA,.pointer "B" stepB] =>
    some ⟨index,lo,hi,kk,left,right,ma,mb,stepA,stepB⟩
  | _ => none

structure Tile where
  geometry : Geometry
  rm : Expr
  rn : Expr
  rk : Expr
  stepA : Expr
  stepB : Expr
  pointerA : Pointer
  pointerB : Pointer
  zeroM : Expr
  zeroN : Expr
  loop : DotLoop
  deriving Repr

structure Mac where
  tile : Expr
  core : Tile
  terminal : Test
  output : Pointer
  mask : Test
  slots : Expr
  scratch : Pointer
  deriving Repr

structure Full where
  tile : Expr
  core : Tile
  rematM : Expr
  rematN : Expr
  output : Pointer
  mask : Test
  deriving Repr

structure First where
  pid : Expr
  start : Expr
  stop : Expr
  condition : Test
  finish : Expr
  advance : Expr
  deriving Repr

structure Reduce where
  tile : Expr
  geometry : Geometry
  rm : Expr
  rn : Expr
  offsets : Expr
  mask : Test
  index : String
  lo : Expr
  hi : Expr
  pid : Expr
  begin : Expr
  finish : Expr
  condition : Test
  slots : Expr
  load : Pointer
  output : Pointer
  atomicMask : Test
  deriving Repr

structure Program where
  first : First
  mac : Mac
  full : Full
  reduce : Reduce
  deriving Repr

structure Sources where
  swizzle : List Statement
  linear : List Statement
  mac : List Statement
  first : List Statement
  full : List Statement
  reduce : List Statement
  deriving Repr

/-- This reader checks supported instruction topology. Loop bounds, branches,
addresses, masks, assignment RHSs and mapping expressions are source data. -/
def compileMac (s : Sources) : Option Mac := do
  match s.mac with
  | [.integer "tile_id" tile,geo,
     .integer "rm" rm,.integer "rn" rn,.integer "rk" rk,
     .integer "step_a" sa,.integer "step_b" sb,
     .pointer "A" pa,.pointer "B" pb,.zeros "acc" zm zn,loop,
     .branch terminal
       [.pointer "C_" pc,.store (.variable "C_") "acc" (some mask)]
       [.integer "slots" slots,.store ps "acc" none]] =>
    let geo ← geometry s.swizzle s.linear geo
    let loop ← dotLoop loop
    some ⟨tile,⟨geo,rm,rn,rk,sa,sb,pa,pb,zm,zn,loop⟩,terminal,pc,mask,slots,ps⟩
  | _ => none

def compileFull (s : Sources) : Option Full := do
  match s.full with
  | [.integer "tile_id" tile,geo,
     .integer "rm" rm,.integer "rn" rn,.integer "rk" rk,
     .integer "step_a" sa,.integer "step_b" sb,
     .pointer "A" pa,.pointer "B" pb,.zeros "acc" zm zn,loop,.half "acc",
     .integer "rm" rem,.integer "rn" ren,
     .pointer "C" pc,.store (.variable "C") "acc" (some mask)] =>
    let geo ← geometry s.swizzle s.linear geo
    let loop ← dotLoop loop
    some ⟨tile,⟨geo,rm,rn,rk,sa,sb,pa,pb,zm,zn,loop⟩,rem,ren,pc,mask⟩
  | _ => none

def compileFirst (s : Sources) : Option First :=
  match s.first with
  | [.integer "pid" pid,.integer "start_iter" start,.integer "last_iter" stop,
     .loop test [.integer "end_iter" finish,.macCall,.integer "start_iter" advance]] =>
    some ⟨pid,start,stop,test,finish,advance⟩
  | _ => none

def compileReduce (s : Sources) : Option Reduce := do
  match s.reduce with
  | [.integer "tile_id" tile,geo,.integer "rm" rm,.integer "rn" rn,
     .integer "offsets" offsets,.boolean "mask" mask,
     .range index lo hi [.integer "pid" pid,.integer "begin" begin,.integer "end" finish,
       .branch condition [.integer "slots" slots,.load "contribution" load none,
         .atomic output "contribution" atomicMask] []]] =>
    let geo ← geometry s.swizzle s.linear geo
    some ⟨tile,geo,rm,rn,offsets,mask,index,lo,hi,pid,begin,finish,condition,slots,load,output,atomicMask⟩
  | _ => none

def compile (s : Sources) : Option Program := do
  return ⟨← compileFirst s,← compileMac s,← compileFull s,← compileReduce s⟩

end VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
