import bench.tritonbench_g.spinning_lock_reduction.HostSource

/-! Full admitted source controller: signed64 pointers, unsigned64 role/group
conditions, two nested while loops, nine rounds, current lock reads, persistent
FP32 accumulators, source masks and both phase guards. Floating instructions
remain generic zero/add/load/store operations; there is no mathematical result
or correctness theorem in this reader. False view flags are established by the
public Python guards; the separate view adapter retains historical encodings.
-/
noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Wide
open VeriTile.Bench.TritonBenchG.SpinningLock

abbrev I64 := BitVec 64

def row64 (h : HostArgs) (p lane : Nat) : I64 :=
  (BitVec.ofNat 64 p).sdiv (BitVec.ofNat 64 h.numSms) * BitVec.ofNat 64 h.bm +
    BitVec.ofNat 64 (lane / h.bn)

def col64 (h : HostArgs) (p lane : Nat) : I64 :=
  (BitVec.ofNat 64 p).srem (BitVec.ofNat 64 h.numSms) * BitVec.ofNat 64 h.bn +
    BitVec.ofNat 64 (lane % h.bn)

def partialOffset64 (h : HostArgs) (q lane : Nat) : Int :=
  (BitVec.ofNat 64 q * BitVec.ofNat 64 h.bm * BitVec.ofNat 64 h.bn).toInt +
    (BitVec.ofNat 64 (lane / h.bn) * BitVec.ofNat 64 h.bn).toInt +
    (BitVec.ofNat 64 (lane % h.bn)).toInt

def outputOffset64 (h : HostArgs) (p lane : Nat) : Int :=
  (row64 h p lane * BitVec.ofNat 64 h.strideM).toInt +
    (col64 h p lane * BitVec.ofNat 64 h.strideN).toInt

def outputMask64 (h : HostArgs) (p lane : Nat) : Bool :=
  (row64 h p lane).slt (BitVec.ofNat 64 h.m) &&
    (col64 h p lane).slt (BitVec.ofNat 64 h.n)

inductive Phase where
  | branch | scan (next : Nat) | poll (next : Nat) | gather (next lane : Nat)
  | publish (lane : Nat) | release | output (lane : Nat) | done | fault
  deriving DecidableEq, Repr

structure Control where
  round : Nat := 1
  phase : Phase := .branch
  deriving DecidableEq, Repr

structure State (V : Type) where
  values : VeriTile.Bench.TritonBenchG.SpinningLock.State V
  pc : Nat → Control

def lanes (h : HostArgs) := h.bm * h.bn
def pAddress (h : HostArgs) (q lane : Nat) : Int := (q * h.bm * h.bn + lane : Nat)
def cAddress (h : HostArgs) (p lane : Nat) : Int :=
  ((lane / h.bn) * h.strideM + (p * h.bn + lane % h.bn) * h.strideN : Nat)
def mask (h : HostArgs) (p lane : Nat) : Bool :=
  decide (lane / h.bn < h.m ∧ p * h.bn + lane % h.bn < h.n)
def reducer (h : HostArgs) (p : Nat) : Bool := decide (p % h.groupK = 0)
def continues (h : HostArgs) (p q : Nat) : Bool := decide (q < p + h.groupK ∧ q < h.numSms)

def initial (b : Backend V) (mem : Memory V) : State V :=
  ⟨VeriTile.Bench.TritonBenchG.SpinningLock.initial b mem, fun _ => {}⟩

/-- Exact source phases: nine rounds, both whiles, CAS old-word polling,
persistent accumulator and locks, all P lanes, and the masked strided C store.
The block barrier is a synchronization boundary before `.release`; it has no
additional memory/value action in this all-lane SC projection. -/
def decode (h : HostArgs) (p : Nat) (s : State V) : Event × Control :=
  let ctl := s.pc p
  if p ≥ h.numSms then (.silent, ctl) else
  match ctl.phase with
  | .branch =>
    if ctl.round ≥ 10 then (.silent, { ctl with phase := .done }) else
    if h.groupK = 0 then (.silent, { ctl with phase := .fault }) else
    (.silent, { ctl with phase := if reducer h p then .scan (p + 1) else .publish 0 })
  | .scan next =>
    (.silent, { ctl with phase := if continues h p next then .poll next else .output 0 })
  | .poll next =>
    (.poll (next : Int), { ctl with phase :=
      if (s.values.mem.locks (next : Int) == 1) then .gather next 0 else .poll next })
  | .gather next lane =>
    if lane < lanes h then (.add p lane (pAddress h next lane), { ctl with phase := .gather next (lane + 1) })
    else (.silent, { ctl with phase := .scan (next + 1) })
  | .publish lane =>
    if lane < lanes h then (.publish p lane (pAddress h p lane), { ctl with phase := .publish (lane + 1) })
    else (.silent, { ctl with phase := .release })
  | .release => (.release (p : Int), { ctl with phase := .output 0 })
  | .output lane =>
    if lane < lanes h then
      (if mask h p lane then .output p lane (cAddress h p lane) else .silent,
       { ctl with phase := .output (lane + 1) })
    else (.silent, { round := ctl.round + 1, phase := .branch })
  | .done | .fault => (.silent, ctl)

def step (b : Backend V) (h : HostArgs) (p : Nat) (s : State V) : Event × State V :=
  let (event, ctl) := decode h p s
  (event, ⟨execute b event s.values, put s.pc p ctl⟩)

def run (b : Backend V) (h : HostArgs) : List Nat → State V → List Event × State V
  | [], s => ([], s)
  | p :: ps, s =>
    let (event, next) := step b h p s
    let (events, final) := run b h ps next
    (event :: events, final)

def allDone (h : HostArgs) (s : State V) : Bool :=
  (List.range h.numSms).all fun p => (s.pc p).phase == .done

def reducer64 (h : HostArgs) (p : Nat) : Bool :=
  (BitVec.ofNat 64 p).umod (BitVec.ofNat 64 h.groupK) == 0

def continues64 (h : HostArgs) (p q : Nat) : Bool :=
  (BitVec.ofNat 64 q).ult (BitVec.ofNat 64 p + BitVec.ofNat 64 h.groupK) &&
    (BitVec.ofNat 64 q).slt (BitVec.ofNat 64 h.numSms)

def nativeDecode (h : HostArgs) (p : Nat) (s : State V) : Event × Control :=
  let ctl := s.pc p
  if p ≥ h.numSms then (.silent, ctl) else
  match ctl.phase with
  | .branch =>
    if ctl.round ≥ 10 then (.silent, { ctl with phase := .done }) else
    (.silent, { ctl with phase := if reducer64 h p then .scan ((BitVec.ofNat 64 p + 1).toNat) else .publish 0 })
  | .scan next =>
    (.silent, { ctl with phase := if continues64 h p next then .poll next else .output 0 })
  | .poll next =>
    (.poll ((BitVec.ofNat 64 next).toInt), { ctl with phase :=
      if (s.values.mem.locks (BitVec.ofNat 64 next).toInt == 1) then .gather next 0 else .poll next })
  | .gather next lane =>
    if lane < lanes h then
      (.add p lane (partialOffset64 h next lane), { ctl with phase := .gather next (lane + 1) })
    else (.silent, { ctl with phase := .scan ((BitVec.ofNat 64 next + 1).toNat) })
  | .publish lane =>
    if lane < lanes h then
      (.publish p lane (partialOffset64 h p lane), { ctl with phase := .publish (lane + 1) })
    else (.silent, { ctl with phase := .release })
  | .release => (.release ((BitVec.ofNat 64 p).toInt), { ctl with phase := .output 0 })
  | .output lane =>
    if lane < lanes h then
      (if outputMask64 h p lane then .output p lane (outputOffset64 h p lane) else .silent,
       { ctl with phase := .output (lane + 1) })
    else (.silent, { round := ctl.round + 1, phase := .branch })
  | .done | .fault => (.silent, ctl)

def nativeStep (b : Backend V) (h : HostArgs) (p : Nat) (s : State V) : Event × State V :=
  let (event, ctl) := nativeDecode h p s
  (event, ⟨execute b event s.values, put s.pc p ctl⟩)

def nativeRun (b : Backend V) (h : HostArgs) : List Nat → State V → List Event × State V
  | [], s => ([], s)
  | p :: ps, s =>
    let (event, next) := nativeStep b h p s
    let (events, final) := nativeRun b h ps next
    (event :: events, final)

end VeriTile.Bench.Repairs.SpinningLock.Wide

namespace VeriTile.Bench.Repairs.SpinningLock.Phased
open VeriTile.Bench.TritonBenchG.SpinningLock Wide

inductive Stage where | producers | consumers deriving DecidableEq, Repr

def selected (h : HostArgs) : Stage → Nat → Bool
  | .producers, p => !reducer h p
  | .consumers, p => reducer h p

def selected64 (h : HostArgs) : Stage → Nat → Bool
  | .producers, p => !reducer64 h p
  | .consumers, p => reducer64 h p

def activeSchedule (h : HostArgs) (stage : Stage) (xs : List Nat) :=
  xs.filter (fun p => decide (p < h.numSms) && selected h stage p)

def nativeSchedule (h : HostArgs) (stage : Stage) (xs : List Nat) :=
  xs.filter (fun p => decide (p < h.numSms) && selected64 h stage p)

def stageRun (b : Backend V) (h : HostArgs) (stage : Stage)
    (xs : List Nat) (s : Wide.State V) : List Event × Wide.State V :=
  nativeRun b h (nativeSchedule h stage xs) s

end VeriTile.Bench.Repairs.SpinningLock.Phased
end
