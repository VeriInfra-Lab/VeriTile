import bench.tritonbench_g.attn_fwd_triton.Proofs.NativeArithmetic
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.NativeOffsets
open VeriTile.Bench.Repairs.LayerNormFwd.Address
/-- K-scale helper starts at lower/64 and advances once per 64-key block. -/
theorem scale_progression (head count lower iterations : Nat) :
    advance (packedScale head count (lower/64)) (wrap64 1) iterations =
      wrap64 (head*count+(lower+iterations*64)/64) := by
  rw [packedScale_mod,advance_mod]
  congr 1
  simp [Nat.add_div,Nat.not_le.mpr (Nat.mod_lt lower (by decide : 0<64)),Nat.add_assoc]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.NativeOffsets
