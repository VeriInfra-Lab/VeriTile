import VeriTile.Triton.Float.IntegerConversion

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.PtxSigned
open VeriTile.Triton
abbrev Word := FP.Value .fp32

/-! The selected Triton 3.7.1/sm80 source emits `cvt.rzi.s16.f32` and
`st.global.b8`. This is signed-16 saturation followed by a low-byte observation,
not direct saturation to signed int8 and not a total portable LLVM fptosi.

PTX conversion and store rules:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st
Rounding is toward zero, integer conversion clamps to the destination range,
and f32 NaN converts to zero. There is no emitted `.ftz` modifier. A byte store
observes only the low eight bits of the converted signed-16 register.
https://llvm.org/docs/LangRef.html#fptosi-to-instruction
LLVM poison outside its defined conversion domain is not assigned a unique
portable meaning by this fixed native profile. No payload guard is added. -/

def cvtS16 (x : Word) : Int :=
  match x.toRat? with
  | some q => min 32767 (max (-32768) (FP.IntegerConversion.truncate q))
  | none => if x.isNaN then 0 else if x.sign then -32768 else 32767

def byte (x : Word) : BitVec 8 := BitVec.ofInt 8 (cvtS16 x)
def signedByte (x : Word) : Int := (byte x).toInt

@[standalone_spec] def CvtS16Spec (x : Word) (z : Int) : Prop :=
  match x.toRat? with
  | some q => ∃ k, FP.IntegerConversion.TowardZero q k ∧ z = min 32767 (max (-32768) k)
  | none => z = if x.isNaN then 0 else if x.sign then -32768 else 32767

theorem cvtS16_spec (x : Word) : CvtS16Spec x (cvtS16 x) := by
  unfold CvtS16Spec cvtS16
  cases x.toRat? with
  | none => rfl
  | some q => exact ⟨_, FP.IntegerConversion.truncate_spec q, rfl⟩

theorem cvtS16_unique (x : Word) (z : Int) (h : CvtS16Spec x z) : z = cvtS16 x := by
  cases hx : x.toRat? with
  | none => simpa [CvtS16Spec, cvtS16, hx] using h
  | some q =>
    simp only [CvtS16Spec, hx] at h
    obtain ⟨k, hk, hz⟩ := h
    have ht := (FP.IntegerConversion.truncate_unique q k).mp hk
    simpa [cvtS16, hx, ht] using hz

theorem cvtS16_bounds (x : Word) : -32768 ≤ cvtS16 x ∧ cvtS16 x ≤ 32767 := by
  unfold cvtS16
  cases hx : x.toRat? with
  | none => simp only; split_ifs <;> omega
  | some q => simp only; omega

theorem nan (x : Word) (h : x.isNaN = Bool.true) : byte x = 0 := by
  have hx : x.exponent = 255 ∧ x.fraction ≠ 0 := by
    simpa [FP.Value.isNaN] using h
  simp [byte, cvtS16, FP.Value.toRat?, FP.Value.decode, hx.1, hx.2, h]

theorem agrees_on_defined_i8 (x : Word) (z : Int)
    (h : FP.IntegerConversion.toSigned 8 x = some z) :
    byte x = BitVec.ofInt 8 z := by
  obtain ⟨q, hq, hz, hr⟩ := (FP.IntegerConversion.toSigned_spec 8 x z).mp h
  have ht := (FP.IntegerConversion.truncate_unique q z).mp hz
  have bounds : -32768 ≤ z ∧ z ≤ 32767 := by
    norm_num [FP.IntegerConversion.SignedRange] at hr
    omega
  simp [byte, cvtS16, hq, ht, max_eq_right bounds.1, min_eq_right bounds.2]

/-- These boundary values distinguish s16-plus-byte from direct s8 saturation.
They are exact Lean kernel computations, not GPU experiments. -/
theorem boundary_examples :
    signedByte (FP.Value.ofBits .fp32 0x00000000) = 0 ∧
    signedByte (FP.Value.ofBits .fp32 0x80000000) = 0 ∧
    signedByte (FP.Value.ofBits .fp32 0x43000000) = -128 ∧
    signedByte (FP.Value.ofBits .fp32 0xc3010000) = 127 ∧
    signedByte (FP.Value.ofBits .fp32 0x47000000) = -1 ∧
    signedByte (FP.Value.infinity .fp32) = -1 ∧
    signedByte (FP.Value.infinity .fp32 Bool.true) = 0 ∧
    signedByte (FP.Value.qNaN .fp32) = 0 := by
  decide +kernel

#axiomsClean cvtS16_spec
#axiomsClean cvtS16_unique
#axiomsClean cvtS16_bounds
#axiomsClean nan
#axiomsClean agrees_on_defined_i8
#axiomsClean boundary_examples
end VeriTile.Bench.Repairs.QuantizeCopyKv.PtxSigned
