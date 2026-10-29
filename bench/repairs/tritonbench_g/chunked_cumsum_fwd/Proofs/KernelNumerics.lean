import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelScalar
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelABI
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Numeric
open VeriTile.Triton FP Scalar
noncomputable section
abbrev Word := BitVec 32
def value (x : Word) : FP.Value .fp32 := ⟨x⟩

/-- PTX leaves single-precision NaN payloads unspecified. A concrete target's
quiet NaN word is therefore explicit, as in the accepted max-reduction model.
Raw loads retain input payloads. No finite-value restriction is introduced. -/
structure Target where
  canonicalNaN : Word
  isNaN : (value canonicalNaN).isNaN=Bool.true
  quiet : canonicalNaN.toNat / 2^22 % 2=1

/-- A concrete representative target, not a claim about an executed GPU. -/
def representative : Target := ⟨0x7fffffff#32,by decide +kernel,by decide +kernel⟩

def result (target : Target) (x : FP.Value .fp32) : Word :=
  if x.isNaN then target.canonicalNaN else x.bits

/-- The saved accumulation instructions are add.f32, mul.f32, fma.rn.f32:
round-to-nearest-even and gradual underflow, without an implicit FTZ step. -/
def add32 (target : Target) (x y : Word) : Word := result target (FP.add .ieeeValue (value x) (value y))
def mul32 (target : Target) (x y : Word) : Word := result target (FP.mul .ieeeValue (value x) (value y))
def fma32 (target : Target) (x y z : Word) : Word := result target (FP.fma .ieeeValue (value x) (value y) (value z))
def le32 (x y : Word) : Bool := FP.lt .ieeeValue (value x) (value y) || FP.eq .ieeeValue (value x) (value y)

/-- SM80 min.f32 with PropagateNan.NONE: one NaN is ignored, both NaNs
use the target NaN word, and a zero tie chooses negative zero. Subnormals are unflushed. -/
def minimum32 (target : Target) (x y : Word) : Word :=
  if (value x).isNaN then (if (value y).isNaN then target.canonicalNaN else y)
  else if (value y).isNaN then x
  else if (value x).isZero && (value y).isZero then
    (FP.Value.zero .fp32 ((value x).sign || (value y).sign)).bits
  else if FP.lt .ieeeValue (value x) (value y) then x else y

/-- The corresponding max.f32 zero tie chooses positive zero. -/
def maximum32 (target : Target) (x y : Word) : Word :=
  if (value x).isNaN then (if (value y).isNaN then target.canonicalNaN else y)
  else if (value y).isNaN then x
  else if (value x).isZero && (value y).isZero then
    (FP.Value.zero .fp32 ((value x).sign && (value y).sign)).bits
  else if FP.lt .ieeeValue (value y) (value x) then x else y

/-- Exactly the three FP32 literals reached by the quotation. -/
def literal (x : ℝ) : Word := if x=0 then 0 else if x=1 then 0x3f800000#32 else if x=20 then 0x41a00000#32 else 0

/-- Fixed unary compiler primitives. exp2Approx is the saved ex2.approx.f32;
logDevice is the saved scalar log lowering, including its internal ftz FMA
sites. Neither port can depend on shape, pointers, a scan or an output. -/
structure UnaryModels where
  exp2Approx : Word→Word
  logDevice : Word→Word

def referenceModel (target : Target) (u : UnaryModels) (reverseBias : Bool) : NativeModel Word where
  biasReverse := reverseBias
  literal := literal
  cast := fun _ _ x=>x
  add := add32 target
  mul := mul32 target
  maximum := maximum32 target
  exp := fun x=>u.exp2Approx (mul32 target x 0x3fb8aa3b#32)
  log := u.logDevice
  minimum := minimum32 target
  lessEqual := le32
  fma := fma32 target
  unused := 0

structure ScalarPorts where
  add : Word→Word→Word
  mul : Word→Word→Word
  maximum : Word→Word→Word
  minimum : Word→Word→Word
  exp : Word→Word
  log : Word→Word
  lessEqual : Word→Word→Bool
  fma : Word→Word→Word→Word

/-- Elementary scalar conformance only. Minimum and the no-FTZ arithmetic
target are explicit definitions above, not deferred experimental placeholders. -/
structure ScalarLaws (target : Target) (u : UnaryModels) (p : ScalarPorts) : Prop where
  add : ∀ x y,p.add x y=add32 target x y
  mul : ∀ x y,p.mul x y=mul32 target x y
  maximum : ∀ x y,p.maximum x y=maximum32 target x y
  minimum : ∀ x y,p.minimum x y=minimum32 target x y
  exp : ∀ x,p.exp x=u.exp2Approx (mul32 target x 0x3fb8aa3b#32)
  log : ∀ x,p.log x=u.logDevice x
  lessEqual : ∀ x y,p.lessEqual x y=le32 x y
  fma : ∀ x y z,p.fma x y z=fma32 target x y z

def nativeModel (p : ScalarPorts) (reverseBias : Bool) : NativeModel Word where
  biasReverse := reverseBias
  literal := literal
  cast := fun _ _ x=>x
  add := p.add
  mul := p.mul
  maximum := p.maximum
  exp := p.exp
  log := p.log
  minimum := p.minimum
  lessEqual := p.lessEqual
  fma := p.fma
  unused := 0

theorem model_eq (target : Target) (u : UnaryModels) (p : ScalarPorts) (laws : ScalarLaws target u p) (reverseBias : Bool) :
    nativeModel p reverseBias=referenceModel target u reverseBias := by
  have ha := funext (fun x=>funext (laws.add x))
  have hm := funext (fun x=>funext (laws.mul x))
  have hmax := funext (fun x=>funext (laws.maximum x))
  have hmin := funext (fun x=>funext (laws.minimum x))
  have he := funext laws.exp
  have hlog := funext laws.log
  have hle := funext (fun x=>funext (laws.lessEqual x))
  have hf := funext (fun x=>funext (fun y=>funext (laws.fma x y)))
  simp only [nativeModel,referenceModel,ha,hm,hmax,hmin,he,hlog,hle,hf]

theorem clamp_special_values :
    minimum32 representative 0x3f800000#32 0x7fc12345#32=0x3f800000#32 ∧
    minimum32 representative 0x7fc12345#32 0x3f800000#32=0x3f800000#32 ∧
    minimum32 representative 0x7fc12345#32 0x7fc54321#32=representative.canonicalNaN ∧
    minimum32 representative 0 0x80000000#32=0x80000000#32 ∧
    maximum32 representative 0 0x80000000#32=0 ∧
    minimum32 representative 1 0x3f800000#32=1 ∧
    minimum32 representative 0x7fc12345#32 0x7f800000#32=0x7f800000#32 := by decide +kernel

/-- Controls distinguish gradual underflow, one fused rounding, and the NaN
that a padded zero produces when multiplied by an infinite rate. -/
theorem arithmetic_controls :
    mul32 representative 1 0x3f800000#32=1 ∧
    add32 representative 1 1=2 ∧
    fma32 representative 0x3f800001#32 0x3f7ffffe#32 0xbf800000#32=0xa8800000#32 ∧
    add32 representative (mul32 representative 0x3f800001#32 0x3f7ffffe#32) 0xbf800000#32=0 ∧
    mul32 representative 0 0x7f800000#32=representative.canonicalNaN := by decide +kernel

#guard_msgs (drop info) in
#specNonCircular ScalarLaws avoiding [Kernel.launch,Scalar.interpret,Scalar.preparedValue,Scalar.cumulativeValue]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Numeric
