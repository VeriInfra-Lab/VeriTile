import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.rowwise_quantization_triton.Kernels
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton
set_option autoImplicit false

def signedByte (byte : Byte) : Int :=
  if byte.val < 128 then byte.val else (byte.val : Int) - 256

/-- The actual store truncates to the low eight bits. In the signed int8 range
this is precisely the standard signed integer interpretation, including negatives. -/
theorem narrow8_signed (z : Int) (lower : -128 ≤ z) (upper : z < 128) :
    signedByte (narrow8 z) = z := by
  have hnonneg : 0 ≤ z % 256 := Int.emod_nonneg z (by norm_num)
  have hlt : z % 256 < 256 := Int.emod_lt_of_pos z (by norm_num)
  have hn : ((z % 256).toNat : Int) = z % 256 := Int.toNat_of_nonneg hnonneg
  have hnat : (z % 256).toNat < 256 := by omega
  have hd := Int.emod_add_ediv z 256
  simp only [signedByte, narrow8, Nat.mod_eq_of_lt hnat]
  split_ifs <;> omega

#axiomsClean narrow8_signed
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton
