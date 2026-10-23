import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Physical

/-! Supporting proof audits. Public/Tensor entry audits live in ../Audit.lean. -/
#axiomsClean VeriTile.Bench.Repairs.LayerNormWelfold.KernelRealResult.kernel_real
#axiomsClean VeriTile.Bench.Repairs.LayerNormWelfold.KernelNativeResult.kernel_native
#axiomsClean VeriTile.Bench.Repairs.LayerNormWelfold.InputBinding.observed
#axiomsClean VeriTile.Bench.Repairs.LayerNormWelfold.InputBinding.native_address
#axiomsClean VeriTile.Bench.Repairs.LayerNormWelfold.NativeInputs.observed
#axiomsClean VeriTile.Bench.Repairs.LayerNormWelfold.Physical.store_bytes
#axiomsClean VeriTile.Bench.Repairs.LayerNormWelfold.Physical.output_pointers
