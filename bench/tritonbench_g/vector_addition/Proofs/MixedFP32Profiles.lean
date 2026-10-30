import bench.tritonbench_g.vector_addition.Proofs.MixedFP32Source

namespace VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32
open VeriTile.Triton FP.Structural FP.IntegerConversion
open IntegerSource (Canonical)
noncomputable section

/-- All 18 ordered canonical integer/bool and fp32 pairs compute in fp32. -/
theorem canonical_promotion (integerFirst : Bool) (d : Canonical) :
    DTypeScope.promote (if integerFirst then d.dtype else .fp32)
      (if integerFirst then .fp32 else d.dtype) = some .fp32 := by
  cases integerFirst <;> cases d <;> decide

/-- A bool destination retains the fptosi result byte 3. There is no
normalization to one at the store boundary. -/
theorem bool_store_three (us : Nat → Word → Int) (uu : Nat → Word → Nat) :
    convertedCell us uu .boolean (FP.Value.ofBits .fp32 0x40400000) = .mk .int 3 := by
  have h : toSigned 8 (FP.Value.ofBits .fp32 0x40400000) = some 3 := by decide +kernel
  simp [convertedCell, totalSigned, h]

/-- fp32 128 is outside signed8 conversion, including a bool output pointer.
The undefined policy remains visible; modular wrapping is not substituted. -/
theorem bool_store_128_undefined (us : Nat → Word → Int) (uu : Nat → Word → Nat) :
    convertedCell us uu .boolean (FP.Value.ofBits .fp32 0x43000000) =
      .mk .int (us 8 (FP.Value.ofBits .fp32 0x43000000)) := by
  have h : toSigned 8 (FP.Value.ofBits .fp32 0x43000000) = none := by decide +kernel
  simp [convertedCell, totalSigned, h]

#axiomsClean canonical_promotion
#axiomsClean bool_store_three
#axiomsClean bool_store_128_undefined
end
end VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32
