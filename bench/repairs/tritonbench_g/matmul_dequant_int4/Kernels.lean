import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulDequantInt4
set_option maxHeartbeats 4000000

abbrev Word := BitVec 32
inductive Region where
  | a | scale | output | weight | dequant | other (n : Nat)
  deriving DecidableEq, Repr
inductive PackedRegion where
  | weight | zero
  deriving DecidableEq, Repr
structure State (α : Type) where
  values : Region → Nat → α
  packed : PackedRegion → Nat → Word

structure Shape where
  m : Nat
  n : Nat
  k : Nat
  group : Nat
  deriving DecidableEq, Repr
structure Shape.Valid (s : Shape) : Prop where
  mpos : 0<s.m
  npos : 0<s.n
  kpos : 0<s.k
  gpos : 0<s.group
  k8 : 8∣s.k
  n8 : 8∣s.n
  groups : s.group∣s.k
structure Tile where
  bm : Nat
  bn : Nat
  bk : Nat
  gm : Nat
  deriving DecidableEq, Repr
structure Tile.Valid (t : Tile) : Prop where
  bmpos : 0<t.bm
  bnpos : 0<t.bn
  bkpos : 0<t.bk
  gmpos : 0<t.gm
  bk16 : 16∣t.bk

def ceil (a b : Nat) := (a+b-1)/b
def blocksM (s : Shape) (t : Tile) := ceil s.m t.bm
def blocksN (s : Shape) (t : Tile) := ceil s.n t.bn
def blocksK (s : Shape) (t : Tile) := ceil s.k t.bk
def nibble (w : Word) (slot : Nat) : Int :=
  ((w >>> (slot%8*4)) &&& (15 : Word)).toInt
def aAddress (s : Shape) (r k : Nat) := r*s.k+k
def qAddress (s : Shape) (k c : Nat) := (k/8)*s.n+c
def sAddress (s : Shape) (k c : Nat) := (k/s.group)*s.n+c
def zAddress (s : Shape) (k c : Nat) := (k/s.group)*(s.n/8)+c/8
def cAddress (s : Shape) (r c : Nat) := r*s.n+c
def difference (s : Shape) (m : State α) (k c : Nat) :=
  nibble (m.packed .weight (qAddress s k c)) k -
  nibble (m.packed .zero (zAddress s k c)) c

structure Work where
  row : Nat
  col : Nat
  deriving DecidableEq, Repr
def Work.Active (rows columns : Nat) (w : Work) := w.row<rows ∧ w.col<columns
instance (rows columns : Nat) (w : Work) : Decidable (w.Active rows columns) :=
  inferInstanceAs (Decidable (_∧_))

def store (m : State α) (r : Region) (address : Nat) (v : α) : State α :=
  {m with values := fun q a => if q=r ∧ a=address then v else m.values q a}

/-- The standalone source subtracts signed integer nibbles before conversion
and multiplication. FP16 and FP32 are distinct instantiations of α. -/
structure DequantOps (α : Type) where
  fromInt : Int → α
  mul : α → α → α
def dequantValue (s : Shape) (p : DequantOps α) (m : State α) (w : Work) :=
  p.mul (p.fromInt (difference s m w.row w.col))
    (m.values .scale (sAddress s w.row w.col))
def dequantStep (s : Shape) (p : DequantOps α) (w : Work) (m : State α) :=
  store m .dequant (cAddress s w.row w.col) (dequantValue s p m w)
def dequantRun (s : Shape) (p : DequantOps α) : List Work → State α → State α
  | [], m => m
  | w::ws, m => dequantRun s p ws (dequantStep s p w m)

/-- Default LLVM fusion may differ among physical lanes within one tile.
This choice is resolved from compiler metadata/register mapping, never data. -/
inductive Fusion where
  | separate | contracted
  deriving DecidableEq, Repr
structure FusedOps (α β : Type) where
  zeroInput : α
  zeroAcc : β
  fromInt : Int → α
  mul : α → α → α
  sub : α → α → α
  neg : α → α
  fma : α → α → α → α
  mma16 : (Fin 16 → α) → (Fin 16 → α) → β → β
  narrow : β → α

def fusedDequant (p : FusedOps α β) (fusion : Fusion) (q z : Int) (scale : α) :=
  let zero := p.mul (p.fromInt z) scale
  match fusion with
  | .separate => p.sub (p.mul (p.fromInt q) scale) zero
  | .contracted => p.fma (p.fromInt q) scale (p.neg zero)

/-- One source iteration uses one group for all BK lanes. Full-group geometry
will prove this equals the independent per-k group lookup. -/
def fusedGroup (s : Shape) (t : Tile) (noGroups : Bool) (block : Nat) :=
  if noGroups then 0 else block/(s.group/t.bk)
def fusedOperandB (s : Shape) (t : Tile) (p : FusedOps α β)
    (noGroups : Bool) (fusion : Nat → Nat → Fusion) (m : State α)
    (col block lane : Nat) :=
  let k := block*t.bk+lane
  let g := fusedGroup s t noGroups block
  fusedDequant p (fusion k col)
    (nibble (m.packed .weight ((k/8)*s.n+col)) lane)
    (nibble (m.packed .zero (g*(s.n/8)+col/8)) col)
    (m.values .scale (g*s.n+col))

def fusedBlock (s : Shape) (t : Tile) (p : FusedOps α β)
    (noGroups : Bool) (fusion : Nat → Nat → Fusion) (m : State α)
    (w : Work) (block : Nat) (seed : β) : β :=
  (List.range (t.bk/16)).foldl (fun acc chunk =>
    p.mma16
      (fun i => if w.row<s.m then m.values .a (aAddress s w.row (block*t.bk+chunk*16+i)) else p.zeroInput)
      (fun i => fusedOperandB s t p noGroups fusion m w.col block (chunk*16+i)) acc) seed
def fusedValue (s : Shape) (t : Tile) (p : FusedOps α β)
    (noGroups : Bool) (fusion : Nat → Nat → Fusion) (m : State α) (w : Work) :=
  p.narrow ((List.range (blocksK s t)).foldl
    (fun acc block => fusedBlock s t p noGroups fusion m w block acc) p.zeroAcc)
def fusedStep (s : Shape) (t : Tile) (p : FusedOps α β)
    (noGroups : Bool) (fusion : Nat → Nat → Fusion) (w : Work) (m : State α) :=
  store m .output (cAddress s w.row w.col) (fusedValue s t p noGroups fusion m w)
def fusedRun (s : Shape) (t : Tile) (p : FusedOps α β)
    (noGroups : Bool) (fusion : Nat → Nat → Fusion) : List Work → State α → State α
  | [], m => m
  | w::ws, m => fusedRun s t p noGroups fusion ws (fusedStep s t p noGroups fusion w m)

end VeriTile.Bench.Repairs.MatmulDequantInt4
