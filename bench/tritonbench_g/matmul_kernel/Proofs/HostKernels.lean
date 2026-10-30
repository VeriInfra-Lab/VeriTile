import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulKernel
set_option maxHeartbeats 2000000
inductive Region where
  | a | b | c | other (id : Nat)
  deriving DecidableEq, Repr
abbrev Memory (α : Type) := Region → Nat → α
abbrev Work := Nat × Nat
abbrev Program := Nat × Nat
/-- Finite fixed-ISA family observed in default SM90 lowering. There is no
variable-K numerical operation. Operand words enter the TF32 instruction
unchanged, as in the emitted PTX (no invented conversion instruction). -/
inductive Opcode where
  | sync | wg16 | wg32 | wg64 | wg128 | wg256
  deriving DecidableEq, Repr
def Opcode.rows : Opcode → Nat | .sync => 16 | _ => 64
def Opcode.cols : Opcode → Nat
  | .sync => 8 | .wg16 => 16 | .wg32 => 32 | .wg64 => 64 | .wg128 => 128 | .wg256 => 256
theorem Opcode.rows_pos (o : Opcode) : 0<o.rows := by cases o <;> decide
theorem Opcode.cols_pos (o : Opcode) : 0<o.cols := by cases o <;> decide
structure Ops (α : Type) where
  zero : α
  mma : (o : Opcode) → Fin o.rows → Fin o.cols → (Fin 8 → α) → (Fin 8 → α) → α → α
  finish : α → α
structure View where
  rows : Nat
  cols : Nat
  sr : Nat
  sc : Nat
  capacity : Nat
  deriving DecidableEq, Repr
def View.addr (v : View) (r c : Nat) := r*v.sr+c*v.sc
structure View.Valid (v : View) : Prop where
  rowPositive : 0<v.sr
  colPositive : 0<v.sc
  separated : v.cols*v.sc≤v.sr ∨ v.rows*v.sr≤v.sc
  allocation : v.rows=0 ∨ v.cols=0 ∨ (v.rows-1)*v.sr+(v.cols-1)*v.sc<v.capacity
structure Config where
  M : Nat
  N : Nat
  K : Nat
  BM : Nat
  BN : Nat
  BK : Nat
  A : View
  B : View
  C : View
  opcode : Opcode
  deriving DecidableEq, Repr
def ceil (x b : Nat) := (x+b-1)/b
def Config.steps (c : Config) := ceil c.K c.BK
def Config.Active (c : Config) (w : Work) := w.1<c.M ∧ w.2<c.N
instance (c : Config) (w : Work) : Decidable (c.Active w) := inferInstanceAs (Decidable (_ ∧ _))
structure Config.Valid (c : Config) : Prop where
  bm : 0<c.BM
  bn : 0<c.BN
  bk : 0<c.BK
  atoms : c.BK/8*8=c.BK
  aShape : c.A.rows=c.M ∧ c.A.cols=c.K
  bShape : c.B.rows=c.K ∧ c.B.cols=c.N
  cShape : c.C.rows=c.M ∧ c.C.cols=c.N
  aValid : c.A.Valid
  bValid : c.B.Valid
  cValid : c.C.Valid
/-- Source's modulo row/column loads and K masks, reading current memory. -/
def readA {α} (c : Config) (p : Ops α) (m : Memory α) (r k : Nat) :=
  if k<c.K then m .a (c.A.addr (r%c.M) k) else p.zero
def readB {α} (c : Config) (p : Ops α) (m : Memory α) (k n : Nat) :=
  if k<c.K then m .b (c.B.addr k (n%c.N)) else p.zero
def atom {α} (c : Config) (p : Ops α) (m : Memory α) (w : Work) (block j : Nat) (acc : α) :=
  p.mma c.opcode ⟨(w.1%c.BM)%c.opcode.rows,Nat.mod_lt _ c.opcode.rows_pos⟩
    ⟨(w.2%c.BN)%c.opcode.cols,Nat.mod_lt _ c.opcode.cols_pos⟩
    (fun e => readA c p m w.1 (block*c.BK+j*8+e.val))
    (fun e => readB c p m (block*c.BK+j*8+e.val) w.2) acc
def block {α} (c : Config) (p : Ops α) (m : Memory α) (w : Work) (q : Nat) (acc : α) :=
  (List.range (c.BK/8)).foldl (fun z j => atom c p m w q j z) acc
def value {α} (c : Config) (p : Ops α) (m : Memory α) (w : Work) :=
  p.finish ((List.range c.steps).foldl (fun z q => block c p m w q z) p.zero)
def Writes (c : Config) (w : Work) (r : Region) (a : Nat) := r=.c ∧ a=c.C.addr w.1 w.2
/-- One active output store after completing all ordered K atoms. -/
def step {α} (c : Config) (p : Ops α) (w : Work) (m : Memory α) : Memory α :=
  fun r a => if r=.c ∧ a=c.C.addr w.1 w.2 then value c p m w else m r a
def execute {α} (c : Config) (p : Ops α) : List Work → Memory α → Memory α
  | [],m => m
  | w::ws,m => execute c p ws (step c p w m)
def programs (c : Config) := (List.range (ceil c.M c.BM)) ×ˢ (List.range (ceil c.N c.BN))
def lanes (c : Config) := (List.range c.BM) ×ˢ (List.range c.BN)
def workOf (c : Config) (pid : Program) (lane : Work) : Work :=
  (pid.1*c.BM+lane.1,pid.2*c.BN+lane.2)
def programWork (c : Config) (pid : Program) :=
  ((lanes c).map (workOf c pid)).filter (fun w => decide (c.Active w))
def schedule (c : Config) := (programs c).flatMap (programWork c)
def run {α} (c : Config) (p : Ops α) (m : Memory α) := execute c p (schedule c) m
end VeriTile.Bench.Repairs.MatmulKernel
