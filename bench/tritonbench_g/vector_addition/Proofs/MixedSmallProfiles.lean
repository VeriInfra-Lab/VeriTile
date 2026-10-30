import bench.tritonbench_g.vector_addition.Proofs.MixedSmallSource

namespace VeriTile.Bench.TritonBenchG.VectorAddition.MixedSmall
open VeriTile.Triton FP.Structural FP.IntegerConversion
open IntegerSource (Canonical)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000

def FloatStorage.canonical : FloatStorage → DTypeScope.DType
  | .fp16 => .fp16 | .bf16 => .bf16

def FloatStorage.promoted : FloatStorage → DTypeScope.DType
  | .fp16 => .fp16 | .bf16 => .fp32

/-- All 36 ordered canonical profiles use their observed promotion dtype. -/
theorem canonical_promotion (f : FloatStorage) (integerFirst : Bool) (d : Canonical) :
    DTypeScope.promote (if integerFirst then d.dtype else f.canonical)
      (if integerFirst then f.canonical else d.dtype) = some f.promoted := by
  cases f <;> cases integerFirst <;> cases d <;> decide

/-- Own-width normalization precedes target rounding. 2049 rounds to the even
binary16 endpoint 2048, whereas the bf16 mixed route rounds to binary32. -/
theorem direct_integer_rounding_vectors :
    integerRounded .ieeeValue .fp16 (.signed 32) 2049 = FP.Value.ofBits .fp32 0x6800 ∧
    integerRounded .ieeeValue .bf16 (.signed 32) 2049 = FP.Value.ofBits .fp32 0x45001000 ∧
    integerRounded .ieeeValue .fp16 (.signed 8) 255 = FP.Value.ofBits .fp32 0xbc00 ∧
    integerRounded .ieeeValue .fp16 (.unsigned 8) 255 = FP.Value.ofBits .fp32 0x5bf8 := by
  decide +kernel

/-- Half arithmetic rounds before extension; bf16 mixed arithmetic rounds in
binary32. The two routes preserve their different integer-store results. -/
theorem arithmetic_and_extension_vectors :
    conversionInput .fp16 (addTarget .ieeeValue .fp16
      (FP.Value.ofBits .fp32 0x6800) (FP.Value.ofBits .fp32 0x3c00)) =
      FP.Value.ofBits .fp32 0x45000000 ∧
    addTarget .ieeeValue .bf16 (FP.Value.ofBits .fp32 0x45000000)
      (extendInput .bf16 (FP.Value.ofBits .fp32 0x3f80)) =
      FP.Value.ofBits .fp32 0x45001000 ∧
    narrowBFloat .ieeeValue (FP.Value.ofBits .fp32 0x45001000) =
      FP.Value.ofBits .fp32 0x4500 ∧
    widenHalf (FP.Value.ofBits .fp32 0x0001) = FP.Value.ofBits .fp32 0x33800000 ∧
    widenBFloat (FP.Value.ofBits .fp32 0x0001) = FP.Value.ofBits .fp32 0x00010000 := by
  decide +kernel

/-- A bool destination stores the raw signed8 result byte, after half widening. -/
theorem bool_store_three (us : Nat → Word → Int) (uu : Nat → Word → Nat) :
    convertedCell us uu .boolean (conversionInput .fp16 (FP.Value.ofBits .fp32 0x4200)) =
      .mk .int 3 := by
  have h : toSigned 8 (conversionInput .fp16 (FP.Value.ofBits .fp32 0x4200)) = some 3 := by
    decide +kernel
  simp [convertedCell, MixedFP32.convertedCell, totalSigned, h]

/-- Infinity from half overflow stays outside the integer-conversion domain. -/
theorem half_overflow_undefined :
    toSigned 32 (conversionInput .fp16
      (integerRounded .ieeeValue .fp16 (.signed 32) 65520)) = none := by
  decide +kernel

#axiomsClean canonical_promotion
#axiomsClean direct_integer_rounding_vectors
#axiomsClean arithmetic_and_extension_vectors
#axiomsClean bool_store_three
#axiomsClean half_overflow_undefined
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.MixedSmall
