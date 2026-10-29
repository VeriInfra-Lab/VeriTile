import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.PrimitiveNumerics

namespace VeriTile.Bench.Repairs.DiagSsmTriton.SoftwareConstants
open Kernels PrimitiveNumerics
set_option autoImplicit false
noncomputable section

theorem zero_decode (config : VeriTile.Triton.FP.Config) : ((software config).zero).toRat? = some 0 := by
  change (VeriTile.Triton.FP.Value.ofBits .fp32 0).toRat? = some 0
  norm_num [VeriTile.Triton.FP.Value.toRat?, VeriTile.Triton.FP.Value.decode,
    VeriTile.Triton.FP.Value.ofBits, VeriTile.Triton.FP.Value.exponent,
    VeriTile.Triton.FP.Value.fraction, VeriTile.Triton.FP.Value.raw,
    VeriTile.Triton.FP.Value.sign, VeriTile.Triton.FP.Format.width,
    VeriTile.Triton.FP.Format.fractionBits]

theorem zero_finite (config : VeriTile.Triton.FP.Config) : PrimitiveNumerics.Finite (software config).zero := by
  simp [PrimitiveNumerics.Finite, zero_decode]

theorem zero_real (config : VeriTile.Triton.FP.Config) : real (software config).zero = 0 := by
  simp [real, zero_decode]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SoftwareConstants
