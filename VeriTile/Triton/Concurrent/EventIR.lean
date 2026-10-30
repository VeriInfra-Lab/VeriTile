import Mathlib.Data.BitVec

/-! Closed scalar events for source-decoded concurrent kernels.

The only parameterized operations are scalar arithmetic primitives. Event
payloads are syntax, not functions from memory to values. Loads, atomic read /
modify / write, and returned old words are interpreted against the current
state by the fixed evaluator below. Region numbers are declared by each IO;
addresses are signed element offsets, so negative accesses remain observable.
-/
namespace VeriTile.Triton.Concurrent

structure Backend (V : Type) where
  zero : V
  add : V → V → V
  mma8 : (Fin 8 → V) → (Fin 8 → V) → V → V
  halfRoundTrip : V → V

structure State (V : Type) where
  input : Nat → Int → V
  mem : Nat → Int → V
  words : Nat → Int → BitVec 32
  registers : Nat → Nat → V
  returned : Nat → Nat → BitVec 32

def update {α β : Type} [DecidableEq α] (f : α → β) (key : α) (value : β) : α → β :=
  fun index => if index = key then value else f index

inductive Expr where
  | zero
  | input (channel : Nat) (index : Int)
  | load (region : Nat) (address : Int)
  | register (thread lane : Nat)
  | add (left right : Expr)
  | mma8 (left right : Fin 8 → Expr) (seed : Expr)
  | halfRoundTrip (value : Expr)

def Expr.eval (b : Backend V) (s : State V) : Expr → V
  | .zero => b.zero
  | .input c i => s.input c i
  | .load r a => s.mem r a
  | .register p l => s.registers p l
  | .add a c => b.add (a.eval b s) (c.eval b s)
  | .mma8 a c seed => b.mma8 (fun i => (a i).eval b s)
      (fun i => (c i).eval b s) (seed.eval b s)
  | .halfRoundTrip x => b.halfRoundTrip (x.eval b s)

inductive Event where
  | store (region : Nat) (address : Int) (value : Expr)
  | assign (thread lane : Nat) (value : Expr)
  | atomicAdd (region : Nat) (address : Int) (value : Expr)
  | compareExchange (region : Nat) (address : Int) (compare replace : BitVec 32)
      (thread destination : Nat)
  | exchange (region : Nat) (address : Int) (value : BitVec 32)
  | readOnly (value : Expr)
  | barrier
  | silent

/-- CAS reads once from the current word heap, returns that old word, and
writes the replacement exactly when the comparison succeeds. -/
def execute (b : Backend V) (event : Event) (s : State V) : State V :=
  match event with
  | .store r a value =>
      {s with mem := update s.mem r (update (s.mem r) a (value.eval b s))}
  | .assign p l value =>
      {s with registers := update s.registers p (update (s.registers p) l (value.eval b s))}
  | .atomicAdd r a value =>
      {s with mem := update s.mem r (update (s.mem r) a (b.add (s.mem r a) (value.eval b s)))}
  | .compareExchange r a compare replace p destination =>
      let old := s.words r a
      { s with
        words := update s.words r (update (s.words r) a (if old == compare then replace else old))
        returned := update s.returned p (update (s.returned p) destination old) }
  | .exchange r a value =>
      {s with words := update s.words r (update (s.words r) a value)}
  | .readOnly _ | .barrier | .silent => s

/-- A tape preserves its supplied source event order; no store/atomic results
are precomputed. Interleaving controllers use the same one-event executor. -/
def run (b : Backend V) : List Event → State V → State V
  | [], s => s
  | event :: events, s => run b events (execute b event s)

structure Bounds where
  input : Nat → Nat
  mem : Nat → Nat
  words : Nat → Nat

def InWindow (size : Nat) (address : Int) : Prop := 0 ≤ address ∧ address < size

def Expr.Safe (bounds : Bounds) : Expr → Prop
  | .zero | .register _ _ => True
  | .input c i => InWindow (bounds.input c) i
  | .load r a => InWindow (bounds.mem r) a
  | .add a c => a.Safe bounds ∧ c.Safe bounds
  | .mma8 a c seed => (∀ i, (a i).Safe bounds) ∧ (∀ i, (c i).Safe bounds) ∧ seed.Safe bounds
  | .halfRoundTrip x => x.Safe bounds

def Event.Safe (bounds : Bounds) : Event → Prop
  | .store r a value | .atomicAdd r a value => InWindow (bounds.mem r) a ∧ value.Safe bounds
  | .assign _ _ value | .readOnly value => value.Safe bounds
  | .compareExchange r a _ _ _ _ | .exchange r a _ => InWindow (bounds.words r) a
  | .barrier | .silent => True

def TapeSafe (bounds : Bounds) (events : List Event) : Prop :=
  ∀ event ∈ events, event.Safe bounds

@[simp] theorem compareExchange_old (b : Backend V) (s : State V)
    (r : Nat) (a : Int) (compare replace : BitVec 32) (p destination : Nat) :
    (execute b (.compareExchange r a compare replace p destination) s).returned p destination =
      s.words r a := by simp [execute, update]

@[simp] theorem compareExchange_memory (b : Backend V) (s : State V)
    (r : Nat) (a : Int) (compare replace : BitVec 32) (p destination : Nat) :
    (execute b (.compareExchange r a compare replace p destination) s).words r a =
      if s.words r a == compare then replace else s.words r a := by simp [execute, update]

/-- Regression: a preceding exchange is seen by CAS, including its returned
old value. A failed comparison leaves the current replacement untouched. -/
theorem exchange_then_failed_cas (b : Backend V) (s : State V) (r : Nat) (a : Int)
    (p destination : Nat) :
    let final := run b [.exchange r a 7, .compareExchange r a 1 9 p destination] s
    final.returned p destination = 7 ∧ final.words r a = 7 := by
  simp [run, execute, update]

theorem exchange_then_successful_cas (b : Backend V) (s : State V) (r : Nat) (a : Int)
    (p destination : Nat) :
    let final := run b [.exchange r a 7, .compareExchange r a 7 9 p destination] s
    final.returned p destination = 7 ∧ final.words r a = 9 := by
  simp [run, execute, update]

end VeriTile.Triton.Concurrent
