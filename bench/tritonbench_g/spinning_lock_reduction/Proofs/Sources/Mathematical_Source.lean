import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

/-!
A task-local, explicit sequentially-consistent lane serialization of the entire
frozen spinning_lock_kernel control graph. This is not the shared Triton DSL:
that DSL has no while. A native refinement to this serialization is NOT claimed.
The two nested whiles, nine rounds, failed CAS retries, unreset accumulator and
locks, repeated P writes, all output masks and every fp32 add are represented.
Memory regions P/C/locks are separate, as in the frozen Python launcher.
-/
namespace VeriTile.Bench.TritonBenchG.SpinningLock
open VeriTile.Triton

abbrev I32 := BitVec 32

structure Config where
  grid : Nat
  bm : Nat
  bn : Nat
  numSms : I32
  groupK : I32
  m : I32
  n : I32
  strideM : I32
  strideN : I32
  deriving DecidableEq, Repr

namespace Config

def pid (_c : Config) (p : Nat) : I32 := BitVec.ofNat 32 p
def lanes (c : Config) : Nat := c.bm * c.bn
/-- Exact signed i32 source operations; no unbounded-Nat control substitution. -/
def reducer (c : Config) (p : Nat) : Bool := (c.pid p).srem c.groupK == 0
def continues (c : Config) (p : Nat) (next : I32) : Bool :=
  next.slt (c.pid p + c.groupK) && next.slt c.numSms

def row (c : Config) (p lane : Nat) : I32 :=
  (c.pid p).sdiv c.numSms * BitVec.ofNat 32 c.bm + BitVec.ofNat 32 (lane / c.bn)
def col (c : Config) (p lane : Nat) : I32 :=
  (c.pid p).srem c.numSms * BitVec.ofNat 32 c.bn + BitVec.ofNat 32 (lane % c.bn)
/-- Each tt.addptr sign-extends its own i32 operand. Do not wrap their sum. -/
def pAddr (c : Config) (p : I32) (lane : Nat) : Int :=
  (p * BitVec.ofNat 32 c.bm * BitVec.ofNat 32 c.bn).toInt +
    (BitVec.ofNat 32 (lane / c.bn) * BitVec.ofNat 32 c.bn).toInt +
    (BitVec.ofNat 32 (lane % c.bn)).toInt

def cAddr (c : Config) (p lane : Nat) : Int :=
  (c.row p lane * c.strideM).toInt + (c.col p lane * c.strideN).toInt

def outputMask (c : Config) (p lane : Nat) : Bool :=
  (c.row p lane).slt c.m && (c.col p lane).slt c.n
/-- Undefined signed divide/remainder is a fault, not a hidden input premise. -/
def divisionDefined (c : Config) (p : Nat) : Bool :=
  c.numSms != 0 && c.groupK != 0 &&
    !((c.pid p == 0x80000000) && (c.numSms == 0xffffffff || c.groupK == 0xffffffff))
end Config

structure Backend (V : Type) where
  zero : V
  add : V → V → V

structure Memory (V : Type) where
  partials : Int → V
  output : Int → V
  locks : Int → I32

inductive Phase where
  | branch
  | scan (next : I32)
  | poll (next : I32)
  | gather (next : I32) (lane : Nat)
  | publish (lane : Nat)
  | release
  | output (lane : Nat)
  | done
  | fault
  deriving DecidableEq, Repr

structure Control where
  round : Nat := 1
  phase : Phase := .branch
  deriving DecidableEq, Repr

structure State (V : Type) where
  mem : Memory V
  control : Nat → Control
  acc : Nat → Nat → V

/-- Events have addresses but NO supplied floating result. The independent
reference recomputes its own reads, sums and store payloads. -/
inductive Event where
  | silent
  | add (pid lane : Nat) (address : Int)
  | publish (pid lane : Nat) (address : Int)
  | release (address : Int)
  | poll (address : Int)
  | output (pid lane : Nat) (address : Int)
  deriving DecidableEq, Repr

/-- Exact CAS compare/replacement pair from source: old word is returned even
when the comparison fails. The RMW synchronization event is retained. -/
def cas11 (old : I32) : I32 × I32 := (old, if old == 1 then 1 else old)

def put {α β : Type} [DecidableEq α] (f : α → β) (key : α) (value : β) : α → β :=
  fun x => if x = key then value else f x

def initial (b : Backend V) (mem : Memory V) : State V :=
  ⟨mem, fun _ => {}, fun _ _ => b.zero⟩

/-- Apply a memory/value instruction, separate from the source control decoder. -/
def execute (b : Backend V) (e : Event) (s : State V) : State V :=
  match e with
  | .silent => s
  | .add p l a => { s with acc := put s.acc p (put (s.acc p) l (b.add (s.acc p l) (s.mem.partials a))) }
  | .publish p l a => { s with mem.partials := put s.mem.partials a (s.acc p l) }
  | .release a => { s with mem.locks := put s.mem.locks a 1 }
  | .poll _ => s -- CAS(1,1) retains the stored bits for success AND failure.
  | .output p l a => { s with mem.output := put s.mem.output a (s.acc p l) }

/-- Source control decoder. The CAS old value is read from CURRENT shared memory;
failed polls remain at the same PC. Scheduling a finished/invalid pid is a no-op.
Within a tile, lane order is explicitly row-major; a GPU refinement is separate. -/
def decode (c : Config) (p : Nat) (s : State V) : Event × Control :=
  let ctl := s.control p
  if p ≥ c.grid then (.silent, ctl) else
  match ctl.phase with
  | .branch =>
    if ctl.round ≥ 10 then (.silent, { ctl with phase := .done }) else
    if !c.divisionDefined p then (.silent, { ctl with phase := .fault }) else
    (.silent, { ctl with phase := if c.reducer p then .scan (c.pid p + 1) else .publish 0 })
  | .scan next =>
    (.silent, { ctl with phase := if c.continues p next then .poll next else .output 0 })
  | .poll next =>
    (.poll next.toInt,
      { ctl with phase := if s.mem.locks next.toInt == 1 then .gather next 0 else .poll next })
  | .gather next lane =>
    if lane < c.lanes then
      (.add p lane (c.pAddr next lane), { ctl with phase := .gather next (lane + 1) })
    else (.silent, { ctl with phase := .scan (next + 1) })
  | .publish lane =>
    if lane < c.lanes then
      (.publish p lane (c.pAddr (c.pid p) lane), { ctl with phase := .publish (lane + 1) })
    else (.silent, { ctl with phase := .release })
  | .release => (.release (c.pid p).toInt, { ctl with phase := .output 0 })
  | .output lane =>
    if lane < c.lanes then
      (if c.outputMask p lane then .output p lane (c.cAddr p lane) else .silent,
       { ctl with phase := .output (lane + 1) })
    else (.silent, { round := ctl.round + 1, phase := .branch })
  | .done | .fault => (.silent, ctl)

def step (b : Backend V) (c : Config) (p : Nat) (s : State V) : Event × State V :=
  let (e, ctl) := decode c p s
  (e, { execute b e s with control := put s.control p ctl })

/-- Finite scheduler prefixes preserve blocked states instead of forcing exit. -/
def run (b : Backend V) (c : Config) : List Nat → State V → List Event × State V
  | [], s => ([], s)
  | p :: ps, s =>
    let (e, s') := step b c p s
    let (es, final) := run b c ps s'
    (e :: es, final)

def allDone (c : Config) (s : State V) : Bool :=
  (List.range c.grid).all fun p => (s.control p).phase == .done

end VeriTile.Bench.TritonBenchG.SpinningLock
