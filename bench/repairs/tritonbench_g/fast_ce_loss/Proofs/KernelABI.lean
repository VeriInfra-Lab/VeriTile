import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelSource
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Native

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelABI
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
abbrev Word := BitVec 32

/-- Exact Python scalar tag and payload. Booleans and tensor scalars have no
constructor. Neither float payloads nor integer arguments pass through ℝ when
selecting their native conversion instruction. -/
inductive PythonScalar where
  | integer (value : Int)
  | floating (bits : BitVec 64)
  deriving DecidableEq

def float64Value (bits : BitVec 64) : Option ℝ :=
  let exponent : Nat := (bits.toNat/2^52)%2048
  let fraction : Nat := bits.toNat%2^52
  if exponent=2047 then none else
    let significand : Nat := if exponent=0 then fraction else 2^52+fraction
    let power : Int := (if exponent=0 then -1022 else (exponent : Int)-1023)-52
    some ((if bits.toNat/2^63=0 then 1 else -1)*(significand : ℝ)*(2:ℝ)^power)

def value : PythonScalar → Option ℝ
  | .integer n => some (n : ℝ)
  | .floating bits => float64Value bits

def Valid : PythonScalar → Prop
  | .integer n => -(2^63 : Int)≤n ∧ n<2^63
  | .floating bits => ∃ x, float64Value bits=some x

structure Options (p : Parameters) where
  softcap : PythonScalar
  scaling : PythonScalar
  softcapValid : Valid softcap
  scalingValid : Valid scaling
  softcapValue : value softcap=some p.softcap
  scalingValue : value scaling=some p.scaling
  softcapFlag : p.hasSoftcap=decide (p.softcap≠0)
  scalingFlag : p.hasScaling=decide (p.scaling≠0)

/-- Explicit individual native weak-scalar conversion instructions. Integer
conversion is direct; binary64 conversion retains the entire original payload. -/
structure Conversions where
  integerToFP32 : Int → Word
  binary64ToFP32 : BitVec 64 → Word

def convert (C : Conversions) : PythonScalar → Word
  | .integer n => C.integerToFP32 n
  | .floating bits => C.binary64ToFP32 bits

/-- Per-literal ABI correspondence for enabled source options only. This is
an instruction/conversion profile, with no output, reduction, or kernel-result
field. Flags depend on the Python value even if conversion rounds to zero or
infinity. -/
structure LiteralProfile (p : Parameters) (o : Options p) (C : Conversions)
    (I : VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.Instructions) : Prop where
  softcap : p.hasSoftcap=Bool.true → I.literal (some .fp32) .real p.softcap=convert C o.softcap
  scaling : p.hasScaling=Bool.true → I.literal (some .fp32) .real p.scaling=convert C o.scaling

 theorem zero_cap_disabled (p : Parameters) (o : Options p) (zero : p.softcap=0) :
    p.hasSoftcap=Bool.false := by simp [o.softcapFlag,zero]
 theorem zero_scaling_disabled (p : Parameters) (o : Options p) (zero : p.scaling=0) :
    p.hasScaling=Bool.false := by simp [o.scalingFlag,zero]
 theorem nonzero_cap_enabled (p : Parameters) (o : Options p) (nonzero : p.softcap≠0) :
    p.hasSoftcap=Bool.true := by simp [o.softcapFlag,nonzero]
 theorem nonzero_scaling_enabled (p : Parameters) (o : Options p) (nonzero : p.scaling≠0) :
    p.hasScaling=Bool.true := by simp [o.scalingFlag,nonzero]

#axiomsClean zero_cap_disabled
#axiomsClean zero_scaling_disabled
#axiomsClean nonzero_cap_enabled
#axiomsClean nonzero_scaling_enabled
#specNonCircular LiteralProfile avoiding [forward,chunked,backward,FP.Structural.exec,FP.Structural.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelABI
