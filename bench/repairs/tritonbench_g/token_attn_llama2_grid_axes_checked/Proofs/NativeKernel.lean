import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Native
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Complete

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.KernelNative
open VeriTile.Triton FP.Structural Whole
noncomputable section

def hostScale (D : Nat) : ℝ :=
  if D=16 then 1/4 else if D=32 then 1592262918131443/9007199254740992
  else if D=64 then 1/8 else 1592262918131443/18014398509481984

def scaleWord (D : Nat) : Native.Word :=
  if D=16 then 0x3e800000 else if D=32 then 0x3e3504f3
  else if D=64 then 0x3e000000 else 0x3db504f3

def algebras (trees : Program → Native.Reductions) : Program → Algebra Native.Term :=
  fun pid => Native.algebra (trees pid)

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.KernelNative
