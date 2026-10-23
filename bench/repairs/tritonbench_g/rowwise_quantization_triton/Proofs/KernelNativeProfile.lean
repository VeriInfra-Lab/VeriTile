import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelProfile
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelEncoding

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelNativeProfile
open VeriTile.Triton FP.Structural
noncomputable section

/-- The documented llrint nearest-even rule on defined FP32 inputs; invalid
NaN/infinity/out-of-signed64-range conversions retain an explicit signed64
policy. This never excludes those inputs from the Python/theorem domain. -/
def round64 (undefined : Word → BitVec 64) (x : Word) : BitVec 64 :=
  match x.toRat? with
  | none => undefined x
  | some q =>
      let z := roundEven (q : ℝ)
      if -(2^63 : Int) ≤ z ∧ z < 2^63 then BitVec.ofInt 64 z else undefined x

/-- Native scalar lowering choices, at their actual source widths. These are
primitive model boundaries, not deferred numerical equivalence assumptions.
No field receives a row, a grid, a Tensor, or an output state. The reduction
schedule is the separate payload-independent `Launch.tree`. -/
structure Arithmetic where
  maximum : Word → Word → Word
  divide : Word → Word → Word
  multiply : Word → Word → Word
  undefinedRound : Word → BitVec 64
  half : Word → Half


def llrint (a : Arithmetic) : Word → BitVec 64 := round64 a.undefinedRound

def zero : Word := FP.Value.zero .fp32
def one : Word := FP.Value.ofBits .fp32 0x3f800000
def scale : Word := FP.Value.ofBits .fp32 0x42fe0000

def operations (a : Arithmetic) : Operations Word where
  zero := zero
  one := one
  scale := scale
  abs := FP.Value.abs
  maximum := a.maximum
  isZero := fun x => FP.eq .ieeeValue x zero
  divide := a.divide
  multiply := a.multiply
  llrint := fun x => (llrint a x).toInt
  toHalf := a.half

def algebra (a : Arithmetic) (p : Launch) (rest : Algebra Word) : Algebra Word :=
  KernelProfile.algebra (operations a) (fun x => KernelEncoding.packHalf (a.half x)) p.tree rest

def equal (a : Arithmetic) := KernelProfile.equal (operations a)

/-- Independent source-order native targets, including the zero-denominator
branch. All floating inputs remain Word values; no finite operand predicate. -/
def maximum (a : Arithmetic) (p : Launch) (values : Nat → Word) : Word :=
  p.tree.evaluate a.maximum (fun c => if c < p.columns then (values c).abs else zero)

def quantized (a : Arithmetic) (p : Launch) (values : Nat → Word) (c : Nat) : Byte :=
  let m := maximum a p values
  let denominator := if FP.eq .ieeeValue m zero then one else m
  narrow8 (llrint a (a.multiply scale (a.divide (values c) denominator))).toInt

def storedMaximum (a : Arithmetic) (p : Launch) (values : Nat → Word) : Half :=
  a.half (maximum a p values)

theorem source_literals : zero.toRat? = some 0 ∧ one.toRat? = some 1 ∧ scale.toRat? = some 127 := by
  decide +kernel

#axiomsClean source_literals
#specNonCircular quantized avoiding [RowwiseQuantizationTriton.source, KernelExecution.exec]
#specNonCircular storedMaximum avoiding [RowwiseQuantizationTriton.source, KernelExecution.exec]
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelNativeProfile
