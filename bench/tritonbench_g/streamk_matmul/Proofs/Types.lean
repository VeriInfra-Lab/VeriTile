import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

/-! Task-local source interpreter. Scalar control uses actual signed 32-bit
operations; native division faults are retained. Memory is an unbounded logical
address space, not an allocation-validity theorem. Array instructions are
serialized one lane at a time; arbitrary program schedules remain observable. -/
namespace VeriTile.Bench.TritonBenchG.StreamK
open VeriTile.Triton
abbrev I32 := BitVec 32
abbrev Address := Nat × Int
inductive Param where
 | m | n | k | bm | bn | bk | groupM | strideAM | strideAK | strideBK | strideBN
 | strideCM | strideCN | fullIters | partialIters | itersPerTile | streamTiles
 deriving DecidableEq, Repr
inductive Shape where
 | scalar | m | n | k | mk | kn | mn
 deriving DecidableEq, Repr
inductive Axis where
 | zero | index | row | col | inner
 deriving DecidableEq, Repr
inductive IExpr where
 | lit (v : I32) | param (p : Param) | pid | axis (a : Axis)
 | get (name : String) (index : Axis)
 | add (a b : IExpr) | sub (a b : IExpr) | mul (a b : IExpr)
 | sdiv (a b : IExpr) | srem (a b : IExpr) | min (a b : IExpr) | cdiv (a b : IExpr)
 deriving DecidableEq, Repr
inductive BExpr where
 | lt (a b : IExpr) | eq (a b : IExpr) | ne (a b : IExpr) | gt (a b : IExpr)
 deriving DecidableEq, Repr
inductive Region where | a | b | c | locks deriving DecidableEq, Repr
inductive Pointer where
 | base (r : Region) | get (name : String) | add (base : Pointer) (offset : IExpr)
 deriving DecidableEq, Repr
inductive Condition where
 | pure (test : BExpr)
 | casNe (ptr : Pointer) (compare replace expected : IExpr)
 deriving DecidableEq, Repr
inductive Command where
 | integer (name : String) (shape : Shape) (value : IExpr)
 | pointer (name : String) (shape : Shape) (value : Pointer)
 | zero (name : String) (shape : Shape)
 | load (name : String) (shape : Shape) (ptr : Pointer)
 | mac (name a b : String)
 | half (name input : String)
 | store (shape : Shape) (ptr : Pointer) (value : String)
 | atomicAdd (shape : Shape) (ptr : Pointer) (value : String)
 | exchange (ptr : Pointer) (value : IExpr)
 | branch (test : BExpr) (yes no : List Command)
 | whileLoop (test : Condition) (body : List Command)
 | forLoop (index : String) (start stop : IExpr) (body : List Command)
 | pass
 -- Internal continuations, never source-produced.
 | forNext (index : String) (stop : I32) (body : List Command)
 | loadLane (name : String) (shape : Shape) (ptr : Pointer) (lane : Nat)
 | storeLane (atomic : Bool) (shape : Shape) (ptr : Pointer) (value : String) (lane : Nat)
 deriving Repr
structure Source where
 first : List Command
 fullWave : List Command
 deriving Repr
structure Config where
 bm : Nat
 bn : Nat
 bk : Nat
 param : Param → I32
 base : Region → Address
 firstGrid : Nat
 fullGrid : Nat

namespace Config
def count (c : Config) : Shape → Nat
 | .scalar => 1 | .m => c.bm | .n => c.bn | .k => c.bk
 | .mk => c.bm*c.bk | .kn => c.bk*c.bn | .mn => c.bm*c.bn

def index (c : Config) (s : Shape) (lane : Nat) : Axis → Nat
 | .zero => 0 | .index => lane
 | .row => match s with | .mk => lane/c.bk | .mn => lane/c.bn | _ => lane
 | .col => match s with | .kn | .mn => lane%c.bn | _ => lane
 | .inner => match s with | .mk => lane%c.bk | .kn => lane/c.bn | _ => lane
end Config
structure Control where
 todo : List Command
 ints : String → Nat → Option I32
 pointers : String → Nat → Option Address
 fault : Bool := false

def put {α β : Type} [DecidableEq α] (f : α→β) (k : α) (v : β) : α→β :=
 fun x => if x=k then v else f x
@[simp] theorem put_same {α β : Type} [DecidableEq α] (f : α→β) (k : α) (v : β) : put f k v k=v := by simp [put]
@[simp] theorem put_other {α β : Type} [DecidableEq α] (f : α→β) (k x : α) (v : β) (h:x≠k) : put f k v x=f x := by simp [put,h]

def divDefined (a b : I32) : Bool := b != 0 && !(a == 0x80000000 && b == 0xffffffff)
def evalI (c : Config) (p : Nat) (s : Shape) (lane : Nat) (ctl : Control) : IExpr → Option I32
 | .lit v => some v | .param x => some (c.param x) | .pid => some (BitVec.ofNat 32 p)
 | .axis a => some (BitVec.ofNat 32 (c.index s lane a))
 | .get name a => ctl.ints name (c.index s lane a)
 | .add a b => return (←evalI c p s lane ctl a)+(←evalI c p s lane ctl b)
 | .sub a b => return (←evalI c p s lane ctl a)-(←evalI c p s lane ctl b)
 | .mul a b => return (←evalI c p s lane ctl a)*(←evalI c p s lane ctl b)
 | .min a b => do
   let x ←evalI c p s lane ctl a; let y ←evalI c p s lane ctl b
   return if x.slt y then x else y
 | .sdiv a b => do
   let x ←evalI c p s lane ctl a; let y ←evalI c p s lane ctl b
   if divDefined x y then some (x.sdiv y) else none
 | .srem a b => do
   let x ←evalI c p s lane ctl a; let y ←evalI c p s lane ctl b
   if divDefined x y then some (x.srem y) else none
 | .cdiv a b => do
   let x ←evalI c p s lane ctl a; let y ←evalI c p s lane ctl b
   let z := x+y-1
   if divDefined z y then some (z.sdiv y) else none

def evalB (c : Config) (p : Nat) (ctl : Control) : BExpr → Option Bool
 | .lt a b => return (←evalI c p .scalar 0 ctl a).slt (←evalI c p .scalar 0 ctl b)
 | .gt a b => return (←evalI c p .scalar 0 ctl b).slt (←evalI c p .scalar 0 ctl a)
 | .eq a b => return (←evalI c p .scalar 0 ctl a)==(←evalI c p .scalar 0 ctl b)
 | .ne a b => return (←evalI c p .scalar 0 ctl a)!=(←evalI c p .scalar 0 ctl b)

def evalPointer (c : Config) (p : Nat) (s : Shape) (lane : Nat) (ctl : Control) : Pointer→Option Address
 | .base r => some (c.base r)
 | .get n => ctl.pointers n lane
 | .add ptr off => do
   let a ←evalPointer c p s lane ctl ptr; let o ←evalI c p s lane ctl off
   return (a.1,a.2+o.toInt)

/-- The dot callback receives input tiles, never the accumulator. The source
then separately adds the returned dot value, retaining its rounding point. -/
structure Backend (V : Type) where
 zero : V
 dot : (m n k : Nat) → (Fin (m*k)→V) → (Fin (k*n)→V) → Fin (m*n)→V
 add : V→V→V
 half : V→V
 output : V→V
 atomicAdd : V→V→V

structure Memory (V : Type) where
 data : Address→V
 locks : Address→I32
structure State (V : Type) where
 mem : Memory V
 control : Nat→Control
 regs : Nat→String→Nat→V
inductive Event where
 | silent
 | reset (pid : Nat) (name : String)
 | load (pid : Nat) (name : String) (lane : Nat) (address : Address)
 | mac (pid : Nat) (name a b : String)
 | half (pid : Nat) (name input : String)
 | store (pid : Nat) (name : String) (lane : Nat) (address : Address)
 | atomicAdd (pid : Nat) (name : String) (lane : Nat) (address : Address)
 | exchange (address : Address) (value : I32)
 | cas (address : Address) (compare replace : I32)
 deriving DecidableEq, Repr

def writeAddress : Event→Option Address
 | .store _ _ _ a | .atomicAdd _ _ _ a => some a
 | _ => none

def execute (c : Config) (b : Backend V) (e : Event) (s : State V) : State V :=
 match e with
 | .silent => s
 | .reset p n => {s with regs := put s.regs p (put (s.regs p) n (fun _=>b.zero))}
 | .load p n i a => {s with regs := put s.regs p (put (s.regs p) n (put (s.regs p n) i (s.mem.data a)))}
 | .mac p n a d =>
   let dot := b.dot c.bm c.bn c.bk (fun i=>s.regs p a i) (fun i=>s.regs p d i)
   {s with regs := put s.regs p (put (s.regs p) n (fun i=> if h:i<c.bm*c.bn then b.add (s.regs p n i) (dot ⟨i,h⟩) else s.regs p n i))}
 | .half p n v => {s with regs := put s.regs p (put (s.regs p) n (fun i=>b.half (s.regs p v i)))}
 | .store p n i a => {s with mem.data := put s.mem.data a (b.output (s.regs p n i))}
 | .atomicAdd p n i a => {s with mem.data := put s.mem.data a (b.atomicAdd (s.mem.data a) (b.output (s.regs p n i)))}
 | .exchange a v => {s with mem.locks := put s.mem.locks a v}
 | .cas a cmp rep => {s with mem.locks := put s.mem.locks a (if s.mem.locks a == cmp then rep else s.mem.locks a)}
end VeriTile.Bench.TritonBenchG.StreamK
