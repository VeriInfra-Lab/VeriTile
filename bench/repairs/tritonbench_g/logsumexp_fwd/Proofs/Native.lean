import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Engine
import bench.repairs.tritonbench_g.max_reduction.Proofs.Native
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.LogsumexpFwd.Native
open VeriTile.Triton
noncomputable section
open Classical
abbrev Word := BitVec 32
abbrev Tree := VeriTile.Bench.Repairs.MaxReduction.Tree
namespace Tree
export VeriTile.Bench.Repairs.MaxReduction.Tree (eval leaves)
end Tree

structure Ops where
  multiply : Word → Word → Word
  subtract : Word → Word → Word
  add : Word → Word → Word
  exponential : Word → Word
  logarithm : Word → Word
  maximum : Word → Word → Word
  equal : Word → Word → Bool

/-- Target-local semantics of the unchanged approximate `tl.exp` and `tl.log`
intrinsics are explicit unary binary32 functions, including exceptional words.
They are not replaced with libdevice or correctly-rounded Real transcendental
functions. Binary arithmetic uses fixed RNE binary32 with input/result FTZ disabled. -/
structure Target where
  arithmeticNaN : VeriTile.Bench.Repairs.MaxReduction.Native.Target
  maximum : VeriTile.Bench.Repairs.MaxReduction.Native.Target
  tlExp32 : Word → Word
  tlLog32 : Word → Word

def value (word : Word) : FP.Value .fp32 := ⟨word⟩
/-- The software arithmetic model canonicalizes NaNs; translate that result
to the actual target ABI's quiet NaN word. Do not silently equate a GPU NaN
payload with the software model's chosen payload. -/
def arithmeticWord (target : Target) (word : Word) : Word :=
  if VeriTile.Bench.Repairs.MaxReduction.Native.nan word then target.arithmeticNaN.canonicalNaN else word
def reference (target : Target) : Ops where
  multiply := fun x y => arithmeticWord target (FP.mul .ieeeValue (value x) (value y)).bits
  subtract := fun x y => arithmeticWord target (FP.sub .ieeeValue (value x) (value y)).bits
  add := fun x y => arithmeticWord target (FP.add .ieeeValue (value x) (value y)).bits
  exponential := target.tlExp32
  logarithm := target.tlLog32
  maximum := VeriTile.Bench.Repairs.MaxReduction.Native.maximum target.maximum
  equal := VeriTile.Bench.Repairs.MaxReduction.Native.equal

/-- Seven exact local unary/binary ports, universally quantified over raw
binary32 words. There is no whole-output, reduction, index or memory law. -/
structure LocalObligations (actual : Ops) (target : Target) : Prop where
  mul32 : ∀ x y,actual.multiply x y=(reference target).multiply x y
  sub32 : ∀ x y,actual.subtract x y=(reference target).subtract x y
  add32 : ∀ x y,actual.add x y=(reference target).add x y
  exp32 : ∀ x,actual.exponential x=(reference target).exponential x
  log32 : ∀ x,actual.logarithm x=(reference target).logarithm x
  maxnum32 : ∀ x y,actual.maximum x y=(reference target).maximum x y
  orderedEq32 : ∀ x y,actual.equal x y=(reference target).equal x y

/-- Actual compiler-selected reduction trees are retained independently for
maximum and sum. The ordinary successful compiler profile supplies each lane
exactly once; this is a source/layout boundary, not a numerical assumption. -/
structure Plans where
  maximum : Nat → Tree
  sum : Nat → Tree
  maxLeaves : ∀ width,0<width → (Tree.leaves (maximum width)).Perm (List.range width)
  sumLeaves : ∀ width,0<width → (Tree.leaves (sum width)).Perm (List.range width)

def engine (ops : Ops) (plans : Plans) : Engine Word where
  zero := 0
  negInf := BitVec.ofNat 32 0xff800000
  bits := fun word => word.bits
  multiply := ops.multiply
  subtract := ops.subtract
  add := ops.add
  exponential := ops.exponential
  logarithm := ops.logarithm
  equal := ops.equal
  maximum := fun width values => Tree.eval ops.maximum
    (fun i => if h : i<width then values ⟨i,h⟩ else BitVec.ofNat 32 0xff800000) (plans.maximum width)
  sum := fun width values => Tree.eval ops.add
    (fun i => if h : i<width then values ⟨i,h⟩ else 0) (plans.sum width)

theorem ops_eq (actual : Ops) (target : Target) (ports : LocalObligations actual target) :
    actual=reference target := by
  cases actual with
  | mk multiply subtract add exponential logarithm maximum equal =>
    have hm := funext (fun x => funext (ports.mul32 x))
    have hs := funext (fun x => funext (ports.sub32 x))
    have ha := funext (fun x => funext (ports.add32 x))
    have he := funext ports.exp32
    have hl := funext ports.log32
    have hx := funext (fun x => funext (ports.maxnum32 x))
    have hq := funext (fun x => funext (ports.orderedEq32 x))
    simp only at hm hs ha he hl hx hq
    cases hm;cases hs;cases ha;cases he;cases hl;cases hx;cases hq
    rfl

theorem block_equiv (actual : Ops) (target : Target) (ports : LocalObligations actual target)
    (plans : Plans) (input : Nat → Word) (scale : Word) (row block D B : Nat) (hasScale : Bool) :
    blockValue (engine actual plans) input scale row block D B hasScale=
      blockValue (engine (reference target) plans) input scale row block D B hasScale := by
  rw [ops_eq actual target ports]

#axiomsClean ops_eq
#axiomsClean block_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd.Native
