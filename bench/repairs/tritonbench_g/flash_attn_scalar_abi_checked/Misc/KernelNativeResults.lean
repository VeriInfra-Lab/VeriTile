import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Kernels
import bench.repairs.tritonbench_g.flash_attn.KernelOnly.NativeResult

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.KernelNativeResults
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource FlashAttn.KernelNative
noncomputable section

/-- Complete source execution and independent native instruction observations
over raw layouts and word memory, preserving every exterior cell. The scalar
ABI retains the original raw scalar object/FP32 word interpretation. -/
theorem whole_native (I : Instructions) (x : Layout) (scale : Checked.ScalarABI) (memory : Memory Word) :
    ∃ final,run x scale memory=some final ∧ final=reference x scale memory ∧
      (∀ r off,observe I x final r off=target I x scale memory r off) ∧
      (∀ r off,¬Writes (parameters x) (programs (parameters x)) r off →
        final r off=initialMemory memory r off) :=
  KernelNative.whole I x scale memory

#axiomsClean whole_native
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.KernelNativeResults
