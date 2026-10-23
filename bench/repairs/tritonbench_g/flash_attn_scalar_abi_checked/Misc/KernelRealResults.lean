import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Kernels
import bench.repairs.tritonbench_g.flash_attn.KernelOnly.RealResult

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.KernelRealResults
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section

/-- Raw kernel layout/stride geometry, input memory and scale determine all
output/L values and the complete regional frame. -/
theorem whole_real (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x) (m : Memory ℝ) (scale : ℝ) :
    ∃ final,runGrid KernelReal.algebra KernelReal.dot (parameters x) (some scale) (programs (parameters x)) (liftMemory (Real.lift m))=some final ∧
      (∀ b h row col,(Work.output b h row col).active x.shape →
        final "O" (FlashAttn.address x.o b h row col)=.mk .real (some (Spec.output x m scale b h row col))) ∧
      (∀ b h row,(Work.logsum b h row).active x.shape →
        final "L" (FlashAttn.lAddress x.shape b h row)=.mk .real (some (Spec.lse x m scale b h row))) ∧
      (∀ r off,¬Writes (parameters x) (programs (parameters x)) r off →
        final r off=liftMemory (Real.lift m) r off) ∧
      (∀ r∈(["Q","K","V"] : List RegionName),∀ off,final r off=liftMemory (Real.lift m) r off) :=
  KernelReal.whole x hs ho m scale

#axiomsClean whole_real
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.KernelRealResults
