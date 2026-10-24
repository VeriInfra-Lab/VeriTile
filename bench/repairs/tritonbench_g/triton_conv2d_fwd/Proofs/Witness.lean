import bench.repairs.tritonbench_g.triton_conv2d_fwd.Kernels
namespace VeriTile.Bench.Repairs.Conv2d.Witness
open scoped BigOperators
/-- A legal channel-tail choice under the original omitted-other loads. -/
def tailOriginal : Nat := ∑ k ∈ Finset.range 32,(if k<1 then 1 else 1)*(if k<1 then 1 else 1)
def tailRepaired : Nat := ∑ k ∈ Finset.range 32,(if k<1 then 1 else 0)*(if k<1 then 1 else 0)
theorem channel_tail_witness : tailOriginal=32 ∧ tailRepaired=1 ∧ tailOriginal≠tailRepaired := by decide
/-- Input [1,32,1,1], 3x3 weight, padding one. Only the center tap is spatially
active. Every original inactive input load may independently choose one. -/
def paddedOriginal : Nat := ∑ h ∈ Finset.range 3,∑ w ∈ Finset.range 3,
  ∑ _c ∈ Finset.range 32,(if h=1 ∧ w=1 then 1 else 1)*1
def paddedRepaired : Nat := ∑ h ∈ Finset.range 3,∑ w ∈ Finset.range 3,
  ∑ _c ∈ Finset.range 32,(if h=1 ∧ w=1 then 1 else 0)*1
theorem spatial_padding_witness : paddedOriginal=288 ∧ paddedRepaired=32 ∧ paddedOriginal≠paddedRepaired := by decide
#axiomsClean channel_tail_witness
#axiomsClean spatial_padding_witness
end VeriTile.Bench.Repairs.Conv2d.Witness
