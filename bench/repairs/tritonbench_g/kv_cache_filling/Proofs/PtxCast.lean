import VeriTile.Triton.Float.IntegerConversion

namespace VeriTile.Bench.Repairs.KvCacheFilling.PtxCast
open VeriTile.Triton
abbrev Word := FP.Value .fp32
abbrev Byte := BitVec 8

/-! The selected native lowering is `cvt.rzi.u16.f32` followed by the low
byte observation of `st.global.b8`. This module models that emitted profile,
not a total LLVM `fptoui` instruction and not every Triton backend.

Primary sources:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
`rzi` rounds toward zero; float-to-integer conversion clamps to the destination
range by default; this f32/u16 combination converts NaN to zero. No `.ftz` is
present, so sm80 supports subnormal inputs.
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st
A wider source register stores its low instruction-width bits.
https://llvm.org/docs/LangRef.html#fptoui-to-instruction
LLVM produces poison outside the representable truncated integer range.
The theorem `agrees_on_defined_u8` states the precise overlap. No guard is
added, and no primitive property is postponed to an experiment. -/

def cvtU16 (x : Word) : Nat :=
  match x.toRat? with
  | some q => min 65535 (FP.IntegerConversion.truncate q).toNat
  | none => if x.isNaN || x.sign then 0 else 65535

def byte (x : Word) : Byte := BitVec.ofNat 8 (cvtU16 x)

/-- Independent rounding interval plus unsigned saturation and special-value
clauses for the emitted PTX instruction. -/
@[standalone_spec] def CvtU16Spec (x : Word) (z : Nat) : Prop :=
  match x.toRat? with
  | some q => ∃ k : Int, FP.IntegerConversion.TowardZero q k ∧ z = min 65535 k.toNat
  | none => z = if x.isNaN || x.sign then 0 else 65535

theorem cvtU16_spec (x : Word) : CvtU16Spec x (cvtU16 x) := by
  unfold CvtU16Spec cvtU16
  cases hx : x.toRat? with
  | none => rfl
  | some q => exact ⟨_, FP.IntegerConversion.truncate_spec q, rfl⟩

theorem cvtU16_bound (x : Word) : cvtU16 x < 65536 := by
  unfold cvtU16
  cases hx : x.toRat? with
  | none => simp only; split_ifs <;> omega
  | some q => exact lt_of_le_of_lt (Nat.min_le_left _ _) (by omega)

theorem cvtU16_unique (x : Word) (z : Nat) (h : CvtU16Spec x z) : z = cvtU16 x := by
  cases hx : x.toRat? with
  | none => simpa [CvtU16Spec, cvtU16, hx] using h
  | some q =>
    simp only [CvtU16Spec, hx] at h
    obtain ⟨k, hk, hz⟩ := h
    have ht := (FP.IntegerConversion.truncate_unique q k).mp hk
    simpa [cvtU16, hx, ht] using hz

/-- Every NaN sign/payload maps to zero in this f32/u16 conversion. -/
theorem nan (x : Word) (h : x.isNaN = true) : byte x = 0 := by
  have hx : x.exponent = 255 ∧ x.fraction ≠ 0 := by
    simpa [FP.Value.isNaN] using h
  simp [byte, cvtU16, FP.Value.toRat?, FP.Value.decode, hx.1, hx.2, h]

theorem infinity (x : Word) (h : x.isInf = true) :
    byte x = if x.sign then 0 else 255 := by
  have hx : x.exponent = 255 ∧ x.fraction = 0 := by
    simpa [FP.Value.isInf] using h
  cases hs : x.sign <;>
    simp [byte, cvtU16, FP.Value.toRat?, FP.Value.decode, FP.Value.isNaN, hx.1, hx.2, hs]

theorem finite (x : Word) (q : Rat) (h : x.toRat? = some q) :
    byte x = BitVec.ofNat 8 (min 65535 (FP.IntegerConversion.truncate q).toNat) := by
  simp [byte, cvtU16, h]

/-- The native profile agrees with the shared partial LLVM-like converter
whenever that converter is defined. Outside that domain the PTX profile is
explicitly total, without a claim that LLVM poison has a unique value. -/
theorem agrees_on_defined_u8 (x : Word) (z : Nat)
    (h : FP.IntegerConversion.toUnsigned 8 x = some z) :
    byte x = BitVec.ofNat 8 z := by
  obtain ⟨q, hq, hz, hr⟩ := (FP.IntegerConversion.toUnsigned_spec 8 x z).mp h
  have ht := (FP.IntegerConversion.truncate_unique q (z : Int)).mp hz
  have range : z ≤ 65535 := by
    have upper := hr.2.2
    norm_num at upper
    omega
  simp [byte, cvtU16, hq, ht, Nat.min_eq_right range]

theorem finite_near_zero (x : Word) (q : Rat) (hq : x.toRat? = some q)
    (lower : -1 < q) (upper : q < 1) : byte x = 0 := by
  have rounded : FP.IntegerConversion.TowardZero q 0 := by
    unfold FP.IntegerConversion.TowardZero
    split_ifs with hn <;> norm_num <;> constructor <;> linarith
  have ht := (FP.IntegerConversion.truncate_unique q 0).mp rounded
  simp [byte, cvtU16, hq, ht]

/-- Narrowing after both uint8 operations equals narrowing once at the final
byte store, for arbitrary integer operands. No 4-bit value restriction is used. -/
theorem pack8 (a b : Nat) :
    BitVec.ofNat 8 ((a % 256 + ((b % 256) * 16) % 256) % 256) =
      BitVec.ofNat 8 (a + b * 16) := by
  apply BitVec.eq_of_toNat_eq
  simp [BitVec.toNat_ofNat, Nat.add_mod, Nat.mul_mod]

/-- The observed PTX `shl.b16` / `add.s16` implementation has the same low
byte as the source's two uint8 arithmetic operations. The packed intermediate
has no branch, pointer, or other consumer in `_fill_rows`; only `st.b8` reads it. -/
theorem pack16_low8 (a b : Nat) :
    BitVec.ofNat 8 ((a % 65536 + ((b % 65536) * 16) % 65536) % 65536) =
      BitVec.ofNat 8 (a + b * 16) := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofNat]
  norm_num

/-- Exact examples cover both signed zeros, negative fractions, the u8/u16
boundaries, infinities and NaNs. These are Lean kernel computations. -/
theorem boundary_examples :
    byte (FP.Value.ofBits .fp32 0x00000000) = 0 ∧
    byte (FP.Value.ofBits .fp32 0x80000000) = 0 ∧
    byte (FP.Value.ofBits .fp32 0xbf000000) = 0 ∧
    byte (FP.Value.ofBits .fp32 0x437f8000) = 255 ∧
    byte (FP.Value.ofBits .fp32 0x43800000) = 0 ∧
    byte (FP.Value.ofBits .fp32 0x47800000) = 255 ∧
    byte (FP.Value.infinity .fp32) = 255 ∧
    byte (FP.Value.infinity .fp32 true) = 0 ∧
    byte (FP.Value.qNaN .fp32) = 0 ∧
    byte (FP.Value.ofBits .fp32 1) = 0 := by
  decide +kernel

/-- Task-local callback changes only the actual fp32-to-uint8 conversion.
Every admitted FP32 bit pattern returns a byte. Other conversions keep the
caller's existing implementation and unsupported-operation behavior. -/
noncomputable def backend (rest : FP.Structural.Algebra Word) : FP.Structural.Algebra Word :=
  { rest with floatToNat := fun p src width =>
      if (FP.IntegerConversion.fp32Source p src && decide (width = 8)) then
        Option.some (fun x => Option.some ((byte x).toNat))
      else rest.floatToNat p src width }

theorem backend_u8 (rest : FP.Structural.Algebra Word) (p : Option IndexComputePrecision)
    (src : FloatDType) (h : FP.IntegerConversion.fp32Source p src = true) :
    (backend rest).floatToNat p src 8 = some (fun x => some (byte x).toNat) := by
  simp [backend, h]

#axiomsClean cvtU16_spec
#axiomsClean cvtU16_bound
#axiomsClean cvtU16_unique
#axiomsClean nan
#axiomsClean infinity
#axiomsClean agrees_on_defined_u8
#axiomsClean finite_near_zero
#axiomsClean pack8
#axiomsClean pack16_low8
#axiomsClean boundary_examples
#axiomsClean backend_u8
end VeriTile.Bench.Repairs.KvCacheFilling.PtxCast
