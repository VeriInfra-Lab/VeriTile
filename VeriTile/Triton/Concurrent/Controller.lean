import VeriTile.Triton.Concurrent.EventIR

/-! Closed finite control data over the shared event executor.
One scheduled thread executes one event, after a finite tree of pure tests.
Counters are natural numbers; every source fixed-width operation is explicitly
tagged below. A CAS continuation sees its actual returned word after execution.
No controller, result, safety or event-payload callback is accepted.
-/
namespace VeriTile.Triton.Concurrent

inductive NumExpr where
  | literal (value : Nat)
  | thread
  | local (slot : Nat)
  | add (left right : NumExpr)
  | mul (left right : NumExpr)
  | div (left right : NumExpr)
  | rem (left right : NumExpr)
  | word64 (value : NumExpr)
  | wordAdd (left right : NumExpr)
  | wordMul (left right : NumExpr)
  | signedDiv (left right : NumExpr)
  | signedRem (left right : NumExpr)
  | unsignedDiv (left right : NumExpr)
  | unsignedRem (left right : NumExpr)
  deriving DecidableEq, Repr

def NumExpr.eval (thread : Nat) (locals : Nat → Nat) : NumExpr → Nat
  | .literal n => n
  | .thread => thread
  | .local slot => locals slot
  | .add a b => a.eval thread locals + b.eval thread locals
  | .mul a b => a.eval thread locals * b.eval thread locals
  | .div a b => a.eval thread locals / b.eval thread locals
  | .rem a b => a.eval thread locals % b.eval thread locals
  | .word64 x => (BitVec.ofNat 64 (x.eval thread locals)).toNat
  | .wordAdd a b => (BitVec.ofNat 64 (a.eval thread locals) + BitVec.ofNat 64 (b.eval thread locals)).toNat
  | .wordMul a b => (BitVec.ofNat 64 (a.eval thread locals) * BitVec.ofNat 64 (b.eval thread locals)).toNat
  | .signedDiv a b => ((BitVec.ofNat 64 (a.eval thread locals)).sdiv (BitVec.ofNat 64 (b.eval thread locals))).toNat
  | .signedRem a b => ((BitVec.ofNat 64 (a.eval thread locals)).srem (BitVec.ofNat 64 (b.eval thread locals))).toNat
  | .unsignedDiv a b => ((BitVec.ofNat 64 (a.eval thread locals)).udiv (BitVec.ofNat 64 (b.eval thread locals))).toNat
  | .unsignedRem a b => ((BitVec.ofNat 64 (a.eval thread locals)).umod (BitVec.ofNat 64 (b.eval thread locals))).toNat

inductive AddrExpr where
  | literal (value : Int)
  | signed64 (value : NumExpr)
  | add (left right : AddrExpr)
  deriving DecidableEq, Repr

def AddrExpr.eval (thread : Nat) (locals : Nat → Nat) : AddrExpr → Int
  | .literal n => n
  | .signed64 x => (BitVec.ofNat 64 (x.eval thread locals)).toInt
  | .add a b => a.eval thread locals + b.eval thread locals

inductive Test where
  | constant (value : Bool)
  | natLt (left right : NumExpr)
  | wordEq (left right : NumExpr)
  | signedLt (left right : NumExpr)
  | unsignedLt (left right : NumExpr)
  | returnedEq (destination : Nat) (value : NumExpr)
  | and (left right : Test)
  | or (left right : Test)
  | not (value : Test)
  deriving DecidableEq, Repr

def Test.eval (s : State V) (thread : Nat) (locals : Nat → Nat) : Test → Bool
  | .constant value => value
  | .natLt a b => a.eval thread locals < b.eval thread locals
  | .wordEq a b => BitVec.ofNat 64 (a.eval thread locals) == BitVec.ofNat 64 (b.eval thread locals)
  | .signedLt a b => (BitVec.ofNat 64 (a.eval thread locals)).slt (BitVec.ofNat 64 (b.eval thread locals))
  | .unsignedLt a b => (BitVec.ofNat 64 (a.eval thread locals)).ult (BitVec.ofNat 64 (b.eval thread locals))
  | .returnedEq destination value => s.returned thread destination == BitVec.ofNat 32 (value.eval thread locals)
  | .and a b => a.eval s thread locals && b.eval s thread locals
  | .or a b => a.eval s thread locals || b.eval s thread locals
  | .not x => !x.eval s thread locals

/-- Dynamic address/register templates needed by the supported controller.
K8/half expressions belong to static event tapes, so they are not duplicated. -/
inductive ValueCode where
  | zero
  | load (region : Nat) (address : AddrExpr)
  | register (lane : NumExpr)
  | add (left right : ValueCode)
  deriving DecidableEq, Repr

def ValueCode.instantiate (thread : Nat) (locals : Nat → Nat) : ValueCode → Expr
  | .zero => .zero
  | .load r a => .load r (a.eval thread locals)
  | .register lane => .register thread (lane.eval thread locals)
  | .add a b => .add (a.instantiate thread locals) (b.instantiate thread locals)

inductive EventCode where
  | store (region : Nat) (address : AddrExpr) (value : ValueCode)
  | assign (lane : NumExpr) (value : ValueCode)
  | compareExchange (region : Nat) (address : AddrExpr) (compare replace : NumExpr)
      (destination : Nat)
  | exchange (region : Nat) (address : AddrExpr) (value : NumExpr)
  | barrier
  | silent
  deriving DecidableEq, Repr

def EventCode.instantiate (thread : Nat) (locals : Nat → Nat) : EventCode → Event
  | .store r a value => .store r (a.eval thread locals) (value.instantiate thread locals)
  | .assign lane value => .assign thread (lane.eval thread locals) (value.instantiate thread locals)
  | .compareExchange r a compare replace destination => .compareExchange r (a.eval thread locals)
      (BitVec.ofNat 32 (compare.eval thread locals)) (BitVec.ofNat 32 (replace.eval thread locals)) thread destination
  | .exchange r a value => .exchange r (a.eval thread locals) (BitVec.ofNat 32 (value.eval thread locals))
  | .barrier => .barrier
  | .silent => .silent

inductive Next where
  | goto (label : Nat) (assignments : List (Nat × NumExpr))
  | branch (condition : Test) (yes no : Next)
  deriving DecidableEq, Repr

def Next.eval (s : State V) (thread : Nat) (locals : Nat → Nat) : Next → Nat × (Nat → Nat)
  | .goto label assignments =>
      (label, assignments.foldl (fun result (slot, value) =>
        update result slot (value.eval thread locals)) locals)
  | .branch condition yes no =>
      if condition.eval s thread locals then yes.eval s thread locals else no.eval s thread locals

inductive Code where
  | emit (event : EventCode) (next : Next)
  | branch (condition : Test) (yes no : Code)
  deriving DecidableEq, Repr

def Code.decode (s : State V) (thread : Nat) (locals : Nat → Nat) : Code → Event × Next
  | .emit event next => (event.instantiate thread locals, next)
  | .branch condition yes no =>
      if condition.eval s thread locals then yes.decode s thread locals else no.decode s thread locals

structure Controller where
  threads : Nat
  enabled : Test
  entry : Nat
  initialLocals : List Nat
  code : List Code
  terminal : List Nat
  deriving DecidableEq, Repr

structure ControlState (V : Type) where
  values : State V
  pc : Nat → Nat
  locals : Nat → Nat → Nat

def Controller.initial (program : Controller) (values : State V) : ControlState V :=
  ⟨values, fun _ => program.entry, fun _ slot => program.initialLocals[slot]?.getD 0⟩

def Controller.step (b : Backend V) (program : Controller) (thread : Nat)
    (s : ControlState V) : Option (Event × ControlState V) := do
  let code ← program.code[s.pc thread]?
  let (event, next) := code.decode s.values thread (s.locals thread)
  let values := execute b event s.values
  let (label, locals) := next.eval values thread (s.locals thread)
  some (event, ⟨values, update s.pc thread label, update s.locals thread locals⟩)

/-- Scheduling a disabled/out-of-grid thread is silent without advancing its
PC. A missing instruction is an execution failure, never successful halt. -/
def Controller.run (b : Backend V) (program : Controller) :
    List Nat → ControlState V → Option (List Event × ControlState V)
  | [], s => some ([], s)
  | thread :: schedule, s =>
      if thread ≥ program.threads || !(program.enabled.eval s.values thread (s.locals thread)) then
        program.run b schedule s
      else do
        let (event, next) ← program.step b thread s
        let (events, final) ← program.run b schedule next
        some (event :: events, final)

def Controller.Done (program : Controller) (s : ControlState V) : Prop :=
  ∀ thread, thread < program.threads → program.enabled.eval s.values thread (s.locals thread) = true →
    s.pc thread ∈ program.terminal

end VeriTile.Triton.Concurrent
