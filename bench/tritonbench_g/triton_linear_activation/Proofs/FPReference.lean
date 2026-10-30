import bench.tritonbench_g.triton_linear_activation.Proofs.EventGrid
namespace VeriTile.Bench.Repairs.TritonLinearActivation
open VeriTile.Triton
abbrev Word := FP.Value .fp32

/-- FP32 arithmetic values use all 32 bits. FP16 storage is represented in the
low 16 bits of the same container; those raw containers are decoded before
arithmetic and never mistaken for FP32 subnormal operands. -/
def roundShiftEven (bits shift : Nat) : Nat :=
  let high := bits/2^shift
  let low := bits%2^shift
  high + if 2*low>2^shift ∨ (2*low=2^shift ∧ high%2=1) then 1 else 0

def packHalf (x : Word) : Word :=
  let sign := if x.sign then 32768 else 0
  let e := x.exponent
  let f := x.fraction
  FP.Value.ofBits .fp32 <|
    if e=255 then if f=0 then sign+31744 else 32256
    else if e>142 then sign+31744
    else if e<102 then sign
    else if e<113 then sign+roundShiftEven (8388608+f) (126-e)
    else sign+(e-112)*1024+roundShiftEven f 13

def unpackHalf (x : Word) : Word :=
  let bits := x.raw%65536
  let negative := bits≥32768
  let e := (bits/1024)%32
  let f := bits%1024
  if e=31 then
    if f=0 then FP.Value.infinity .fp32 negative else FP.Value.qNaN .fp32
  else if e=0 then
    FP.round .fp32 FP.Config.ieeeValue
      ((if negative then -1 else 1)*Rat.ofInt f*FP.pow2 (-24)) negative
  else FP.Value.ofBits .fp32 ((if negative then 2147483648 else 0)+(e+112)*8388608+f*8192)

def maxnum (x y : Word) : Word :=
  if x.isNaN then if y.isNaN then FP.Value.qNaN .fp32 else y
  else if y.isNaN then x
  else if x.isZero && y.isZero then FP.Value.zero .fp32 (x.sign && y.sign)
  else if FP.lt FP.Config.ieeeValue x y then y else x

def literalWord : Literal → Word
  | .zero => FP.Value.ofBits .fp32 0x00000000
  | .one => FP.Value.ofBits .fp32 0x3f800000
  | .half => FP.Value.ofBits .fp32 0x3f000000
  | .cubic => FP.Value.ofBits .fp32 0x3d372713
  | .sqrtTwo => FP.Value.ofBits .fp32 0x3fb504f3
  | .sqrtTwoOverPi => FP.Value.ofBits .fp32 0x3f4c422a

/-- Deferred profiles are local scalar libdevice/div.full operations or one
result lane of a named K8/K16 ISA instruction. There is no arbitrary K, matrix,
whole-dot, activation-vector or completed-output parameter. -/
structure LocalProfile where
  divFullF32 : Word → Word → Word
  libdeviceTanhF32 : Word → Word
  libdeviceErfF32 : Word → Word
  instructionLane : (kind : Instruction) → (Fin kind.width → Word) → (Fin kind.width → Word) → Word → Word

noncomputable def fpArithmetic (p : LocalProfile) : Arithmetic Word where
  literal := literalWord
  encode := fun mode => match mode with | .fp32 => id | .autocastFP16 => packHalf
  decode := fun mode => match mode with | .fp32 => id | .autocastFP16 => unpackHalf
  add := FP.add FP.Config.ieeeValue
  mul := FP.mul FP.Config.ieeeValue
  fma := FP.fma FP.Config.ieeeValue
  divFull := p.divFullF32
  maximum := maxnum
  tanh := p.libdeviceTanhF32
  erf := p.libdeviceErfF32
  instruction := p.instructionLane

structure PrimitiveOps where
  packHalf : Word → Word
  unpackHalf : Word → Word
  add : Word → Word → Word
  mul : Word → Word → Word
  fma : Word → Word → Word → Word
  divFull : Word → Word → Word
  maximum : Word → Word → Word
  tanh : Word → Word
  erf : Word → Word
  instruction : (kind : Instruction) → (Fin kind.width → Word) → (Fin kind.width → Word) → Word → Word

def PrimitiveOps.arithmetic (p : PrimitiveOps) : Arithmetic Word where
  literal := literalWord
  encode := fun mode => match mode with | .fp32 => id | .autocastFP16 => p.packHalf
  decode := fun mode => match mode with | .fp32 => id | .autocastFP16 => p.unpackHalf
  add := p.add
  mul := p.mul
  fma := p.fma
  divFull := p.divFull
  maximum := p.maximum
  tanh := p.tanh
  erf := p.erf
  instruction := p.instruction

structure NumericLeaves (actual : PrimitiveOps) (p : LocalProfile) : Prop where
  packHalf : ∀ x,actual.packHalf x=TritonLinearActivation.packHalf x
  unpackHalf : ∀ x,actual.unpackHalf x=TritonLinearActivation.unpackHalf x
  add : ∀ x y,actual.add x y=FP.add FP.Config.ieeeValue x y
  mul : ∀ x y,actual.mul x y=FP.mul FP.Config.ieeeValue x y
  fma : ∀ x y z,actual.fma x y z=FP.fma FP.Config.ieeeValue x y z
  divFull : ∀ x y,actual.divFull x y=p.divFullF32 x y
  maximum : ∀ x y,actual.maximum x y=maxnum x y
  tanh : ∀ x,actual.tanh x=p.libdeviceTanhF32 x
  erf : ∀ x,actual.erf x=p.libdeviceErfF32 x
  instruction : ∀ kind xs ys z,actual.instruction kind xs ys z=p.instructionLane kind xs ys z

noncomputable section
 theorem NumericLeaves.arithmetic_eq (h : NumericLeaves actual p) : actual.arithmetic=fpArithmetic p := by
  have hp := funext h.packHalf
  have hu := funext h.unpackHalf
  have ha := funext fun x => funext (h.add x)
  have hm := funext fun x => funext (h.mul x)
  have hf := funext fun x => funext fun y => funext (h.fma x y)
  have hd := funext fun x => funext (h.divFull x)
  have hx := funext fun x => funext (h.maximum x)
  have ht := funext h.tanh
  have he := funext h.erf
  have hi := funext fun kind => funext fun xs => funext fun ys => funext (h.instruction kind xs ys)
  simp only [PrimitiveOps.arithmetic,fpArithmetic,hp,hu,ha,hm,hf,hd,hx,ht,he,hi]

/-- Equality is word/byte equality, not an assertion that rounded execution is
identical to the independent Real target. Special values remain bit patterns. -/
theorem native_fp_grid (primitive : PrimitiveOps) (p : LocalProfile) (leaves : NumericLeaves primitive p)
    (abi : LocalEventABI Word σ primitive.arithmetic) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (state : σ) (sep : Separate s b) :
    abi.observe ((gridCode s b c mode name).native abi state).2=
      run (fpArithmetic p) s b c mode name (abi.observe state) := by
  rw [native_grid abi s b c mode name state sep]
  exact congrArg (fun model => run model s b c mode name (abi.observe state)) leaves.arithmetic_eq

#axiomsClean NumericLeaves.arithmetic_eq
#axiomsClean native_fp_grid
end
end VeriTile.Bench.Repairs.TritonLinearActivation
