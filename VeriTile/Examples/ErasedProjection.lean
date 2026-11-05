import VeriTile.Triton.Float.Erasure

namespace VeriTile.Triton.ComputeKernel

/-- Erasing dtypes preserves the availability of the algorithm projection. -/
theorem eraseDType_projectable (source : ComputeKernel)
    (h : source.toAlgorithm? = .ok source.toAlgKernel) :
    source.eraseDType.toAlgorithm? = .ok source.eraseDType.toAlgKernel := by
  simp only [eraseDType, h, toAlgorithm?_fromKernelBody, toAlgKernel_fromKernelBody]

end VeriTile.Triton.ComputeKernel
