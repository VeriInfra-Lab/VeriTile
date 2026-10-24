import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Native

namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Floating
open VeriTile.Triton FP.Structural
open Bytes
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

/-- Exact byte packing, independent of the host/kernel implementation. -/
def raw (w : Word) : Nat := ∑ b : Fin 4,(w b).toNat*256^b.val
def word (n : Nat) : Word := fun b=>BitVec.ofNat 8 (n/256^b.val)
def half (n : Nat) : Memory.Half := fun b=>BitVec.ofNat 8 (n/256^b.val)

def roundEven (n denominator : Nat) : Nat :=
  let q:=n/denominator
  let r:=n%denominator
  if 2*r>denominator ∨ (2*r=denominator ∧ q%2=1) then q+1 else q

/-- Exact RN-even binary32-to-binary16 software primitive. Signed zeros,
infinities and subnormals are explicit; NaNs use canonical positive 0x7e00.
Hardware agreement, including NaN policy, is a modeling boundary of this
optional software reference, not a deferred task-level numerical experiment. -/
def convert16 (w : Word) : Memory.Half :=
  let n:=raw w
  let sign:=n/2^31*2^15
  let exponent:=(n/2^23)%256
  let fraction:=n%2^23
  if exponent=255 then half (if fraction=0 then sign+31744 else 32256)
  else if exponent=0 then half sign
  else if exponent>142 then half (sign+31744)
  else if exponent≥113 then half (sign+(exponent-112)*1024+roundEven (2^23+fraction) (2^13)-1024)
  else half (sign+roundEven (2^23+fraction) (2^(126-exponent)))

def multiply32 (x y : Word) : Word :=
  word (FP.Value.raw (FP.mul FP.Config.ieeeValue
    (FP.Value.ofBits .fp32 (raw x)) (FP.Value.ofBits .fp32 (raw y))))

def software : Memory.Numerics := ⟨multiply32,convert16⟩

theorem inverse_bits : raw Memory.inverse=1006699012 := by decide

theorem inverse_rounding :
    FP.Value.raw (FP.round .fp32 FP.Config.ieeeValue (1/127 : Rat))=1006699012 := by decide +kernel

theorem inverse_error : (2113665/268435456 : ℝ)-1/127= -1/34091302912 := by norm_num

theorem inverse_relative_error : ((2113665/268435456 : ℝ)-1/127)/(1/127)= -(1/2^28) := by norm_num

/-- Optional primitive-conformance premises of the retained FP32 reference.
These are not numerical transformation assumptions and are not counted as
deferred experiments for the selected INT8 kernel acceptance. No execution,
indexing, success, frame or whole-output equality is assumed. -/
structure LocalNumerics (ops : Memory.Numerics) (scale x : Word) : Prop where
  first : ops.multiply32 scale x=software.multiply32 scale x
  second : ops.multiply32 (software.multiply32 scale x) Memory.inverse=
    software.multiply32 (software.multiply32 scale x) Memory.inverse
  cast : ops.convert16 (software.multiply32 (software.multiply32 scale x) Memory.inverse)=
    software.convert16 (software.multiply32 (software.multiply32 scale x) Memory.inverse)

theorem compute_reference (ops : Memory.Numerics) (scale x : Word) (laws : LocalNumerics ops scale x) :
    Memory.compute ops scale x=Memory.compute software scale x := by
  unfold Memory.compute
  rw [laws.first,laws.second,laws.cast]

#specNonCircular software avoiding [VeriTile.Triton.FP.Structural.exec]
#axiomsClean inverse_rounding
#axiomsClean inverse_error
#axiomsClean compute_reference
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Floating
