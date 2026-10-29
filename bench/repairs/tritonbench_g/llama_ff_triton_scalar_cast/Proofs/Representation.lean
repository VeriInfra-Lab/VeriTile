import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Words

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Words
open VeriTile.Triton

/-- Signed int8 storage is reinterpreted as precisely the same eight bits,
including both zero signs, subnormals, infinities and every NaN payload. -/
theorem signed_byte_exact (b : Byte) :
    byte (BitVec.ofNat 32 (Float8E5Bits.ofInt b.toInt).val)=b := by
  have h := congrArg BitVec.toNat (BitVec.ofInt_toInt (x := b))
  simp only [BitVec.toNat_ofInt] at h
  apply BitVec.eq_of_toNat_eq
  simp only [byte,Float8E5Bits.ofInt,BitVec.toNat_ofNat]
  norm_num at h ⊢
  rw [h]
  have hb := b.isLt
  omega

#axiomsClean signed_byte_exact
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Words
