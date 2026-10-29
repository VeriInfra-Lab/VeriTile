import bench.repairs.tritonbench_g.log_softmax.Proofs.Graph
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Native

namespace VeriTile.Bench.Repairs.LogSoftmax.Native
open VeriTile.Triton
noncomputable section
open Classical
abbrev Word := BitVec 32

/-- The compiler's local arithmetic operations. Constants, casts and sign-bit
negation are pinned by source/representation; they are not numerical ports. -/
structure Scalars where
  add : Word → Word → Word
  subtract : Word → Word → Word
  multiply : Word → Word → Word
  maximum : Word → Word → Word
  exponential : Word → Word
  logarithm : Word → Word
  fma : Word → Word → Word → Word

abbrev Target := LogsumexpFwd.Native.Target

def reference (target : Target) : Scalars where
  add := (LogsumexpFwd.Native.reference target).add
  subtract := (LogsumexpFwd.Native.reference target).subtract
  multiply := (LogsumexpFwd.Native.reference target).multiply
  maximum := (LogsumexpFwd.Native.reference target).maximum
  exponential := target.tlExp32
  logarithm := target.tlLog32
  fma := fun x y z => LogsumexpFwd.Native.arithmeticWord target
    (FP.fma .ieeeValue (LogsumexpFwd.Native.value x) (LogsumexpFwd.Native.value y) (LogsumexpFwd.Native.value z)).bits

def ops (s : Scalars) : Ops Word where
  zero := 0
  negativeInfinity := BitVec.ofNat 32 0xff800000
  add := s.add
  subtract := s.subtract
  multiply := s.multiply
  maximum := s.maximum
  exp := s.exponential
  log := s.logarithm
  negate := fun x => x ^^^ BitVec.ofNat 32 0x80000000
  fma := s.fma

/-- Historical optional primitive-model refinement, not a numerical-experiment
TODO or a premise of the public source equivalence. None asserts exact Real
arithmetic, a whole reduction, a kernel result or an execution/frame law.
The selected backward schedule explicitly distinguishes separate operations
from either native FMA operand order. -/
structure LocalObligations (actual : Scalars) (target : Target) : Prop where
  add32 : ∀ x y,actual.add x y=(reference target).add x y
  sub32 : ∀ x y,actual.subtract x y=(reference target).subtract x y
  mul32 : ∀ x y,actual.multiply x y=(reference target).multiply x y
  maxnum32 : ∀ x y,actual.maximum x y=(reference target).maximum x y
  exp32 : ∀ x,actual.exponential x=(reference target).exponential x
  log32 : ∀ x,actual.logarithm x=(reference target).logarithm x
  fma32 : ∀ x y z,actual.fma x y z=(reference target).fma x y z

theorem scalars_eq (actual : Scalars) (target : Target) (ports : LocalObligations actual target) :
    actual=reference target := by
  cases actual with
  | mk add subtract multiply maximum exponential logarithm fma =>
    have ha := funext (fun x => funext (ports.add32 x))
    have hs := funext (fun x => funext (ports.sub32 x))
    have hm := funext (fun x => funext (ports.mul32 x))
    have hx := funext (fun x => funext (ports.maxnum32 x))
    have he := funext ports.exp32
    have hl := funext ports.log32
    have hf := funext (fun x => funext (fun y => funext (ports.fma32 x y)))
    simp only at ha hs hm hx he hl hf
    cases ha;cases hs;cases hm;cases hx;cases he;cases hl;cases hf
    rfl

theorem forward_equiv (actual : Scalars) (target : Target) (ports : LocalObligations actual target)
    (plan : Plan N C) (x : Nat → Word) (column : Nat) :
    forward (ops actual) plan x column=forward (ops (reference target)) plan x column := by
  rw [scalars_eq actual target ports]

theorem backward_equiv (actual : Scalars) (target : Target) (ports : LocalObligations actual target)
    (plan : Plan N C) (y g : Nat → Word) (column : Nat) :
    backward (ops actual) plan y g column=backward (ops (reference target)) plan y g column := by
  rw [scalars_eq actual target ports]

#axiomsClean scalars_eq
#axiomsClean forward_equiv
#axiomsClean backward_equiv
end
end VeriTile.Bench.Repairs.LogSoftmax.Native
