import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.bmm_optimized.Sources.Mathematical_RealGrid

namespace VeriTile.Bench.Repairs.BmmOptimized

inductive Tiling where
  | t32x32 | t64x32 | t64x64 | t128x32 | t128x64 | t128x128
  deriving DecidableEq, Repr
structure Config where
  tiling : Tiling
  threeStages : Bool
  deriving DecidableEq, Repr
def Config.tm (c : Config) : Nat := match c.tiling with
  | .t32x32 => 32 | .t64x32 | .t64x64 => 64 | _ => 128
def Config.tn (c : Config) : Nat := match c.tiling with
  | .t32x32 | .t64x32 | .t128x32 => 32
  | .t64x64 | .t128x64 => 64 | _ => 128
def Config.group (c : Config) : Nat := if c.tiling=.t32x32 then 1 else 2
def Config.stages (c : Config) : Nat := if c.threeStages then 3 else 2
def Config.warps (_ : Config) : Nat := 4
def Config.tk (_ : Config) : Nat := 32
theorem config_bounds (c : Config) :
    0<c.tm ∧ c.tm≤128 ∧ 0<c.tn ∧ c.tn≤128 ∧ 0<c.group ∧ c.group≤2 := by
  cases c with | mk t st => cases t <;> simp [Config.tm,Config.tn,Config.group]

structure Shape where
  batch : Nat
  m : Nat
  n : Nat
  k : Nat
  deriving DecidableEq, Repr
def blocks (n tile : Nat) := (n+tile-1)/tile
def Shape.gx (s : Shape) (c : Config) := blocks s.m c.tm
def Shape.gy (s : Shape) (c : Config) := blocks s.n c.tn
def Shape.iterations (s : Shape) := blocks s.k 32
def Shape.a (s : Shape) (b row k : Nat) := (b*s.m+row)*s.k+k
def Shape.b (s : Shape) (b k col : Nat) := (b*s.k+k)*s.n+col
def Shape.o (s : Shape) (b row col : Nat) := (b*s.m+row)*s.n+col
def Shape.size (s : Shape) := s.batch*s.m*s.n
inductive Region where
  | a | b | output | callerA | callerB | other (id : Nat)
  deriving DecidableEq, Repr
abbrev Memory (α : Type) := Region → Nat → α
abbrev Program := Nat×Nat×Nat
abbrev Lane := Nat×Nat
structure Cell where
  batch : Nat
  row : Nat
  column : Nat
  deriving DecidableEq, Repr
def Shape.Active (s : Shape) (x : Cell) : Prop :=
  x.batch<s.batch ∧ x.row<s.m ∧ x.column<s.n
instance (s : Shape) (x : Cell) : Decidable (s.Active x) := inferInstanceAs (Decidable (_∧_∧_))

def rowTile (s : Shape) (c : Config) (pid : Program) :=
  VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile
    pid.1 pid.2.1 (s.gx c) (s.gy c) c.group
def columnTile (s : Shape) (c : Config) (pid : Program) :=
  VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.columnTile
    pid.1 pid.2.1 (s.gx c) (s.gy c) c.group
def cellOf (s : Shape) (c : Config) (pid : Program) (lane : Lane) : Cell :=
  ⟨pid.2.2,rowTile s c pid*c.tm+lane.1,columnTile s c pid*c.tn+lane.2⟩
def programs (s : Shape) (c : Config) : List Program :=
  (List.range (s.gx c)) ×ˢ ((List.range (s.gy c)) ×ˢ (List.range s.batch))
def lanes (c : Config) : List Lane := (List.range c.tm) ×ˢ (List.range c.tn)

/-- The only numerical arithmetic primitive is one ordered scalar multiply-add.
For the real interpretation it is a*b+carry. The selected native interpretation
uses one raw-word PTX fma.rn.f32 per K lane, with no dot-valued callback. -/
structure Scalar (α : Type) where
  zero : α
  fma : α → α → α → α
def aLoad {α : Type} (s : Shape) (p : Scalar α) (m : Memory α) (x : Cell) (k : Nat) :=
  if x.row<s.m ∧ k<s.k then m .a (s.a x.batch x.row k) else p.zero
def bLoad {α : Type} (s : Shape) (p : Scalar α) (m : Memory α) (x : Cell) (k : Nat) :=
  if k<s.k ∧ x.column<s.n then m .b (s.b x.batch k x.column) else p.zero

/-- Explicit source K-lane execution, including zero-filled padded lanes. -/
def laneLoop {α : Type} (s : Shape) (p : Scalar α) (m : Memory α) (x : Cell)
    (start : Nat) : Nat → α → α
  | 0,carry => carry
  | n+1,carry => p.fma (aLoad s p m x (start+n)) (bLoad s p m x (start+n))
      (laneLoop s p m x start n carry)
/-- Source block loop carries its FP32 accumulator across every 32-wide tile. -/
def blockLoop {α : Type} (s : Shape) (p : Scalar α) (m : Memory α) (x : Cell) : Nat → α
  | 0 => p.zero
  | n+1 => laneLoop s p m x (n*32) 32 (blockLoop s p m x n)
def sourceValue {α : Type} (s : Shape) (p : Scalar α) (m : Memory α) (x : Cell) :=
  blockLoop s p m x s.iterations

/-- A source CTA loads from its current memory and scatters only its mask-active
output lanes. Its arithmetic does not read the destination allocation. -/
def store {α : Type} (m : Memory α) (address : Nat) (value : α) : Memory α :=
  fun r a => if r=.output ∧ a=address then value else m r a
def programWrites {α : Type} (s : Shape) (c : Config) (p : Scalar α)
    (pid : Program) (m : Memory α) : List (Nat×α) :=
  ((lanes c).map (cellOf s c pid)).filterMap fun x =>
    if s.Active x then some (s.o x.batch x.row x.column,sourceValue s p m x) else none
def sourceProgram {α : Type} (s : Shape) (c : Config) (p : Scalar α)
    (pid : Program) (m : Memory α) :=
  (programWrites s c p pid m).foldl (fun state w => store state w.1 w.2) m
def sourcePrograms {α : Type} (s : Shape) (c : Config) (p : Scalar α) : List Program → Memory α → Memory α
  | [],m => m
  | pid::rest,m => sourcePrograms s c p rest (sourceProgram s c p pid m)
def source {α : Type} (s : Shape) (c : Config) (p : Scalar α) (m : Memory α) :=
  sourcePrograms s c p (programs s c) m
noncomputable def realScalar : Scalar ℝ := ⟨0,fun a b carry => a*b+carry⟩

end VeriTile.Bench.Repairs.BmmOptimized
