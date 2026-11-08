import bench.examples.OnlineSoftmax.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Mathematical views derived from the current source. The normalizer is a
proof-only prefix; the complete two-pass program remains in Kernels.lean. -/
namespace VeriTile.Bench.Examples.OnlineSoftmax.RealSource
open VeriTile.Triton

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

def stableSoftmaxKernel (x y : RegionName) (B : Nat) : ComputeKernel :=
  (Kernels.stableSoftmaxKernel x y B).eraseDType

def batchSoftmaxKernel (x y : RegionName) (N : Nat) : ComputeKernel :=
  stableSoftmaxKernel x y N

def onlineSoftmaxKernel (x y : RegionName) (N : Nat) : ComputeKernel :=
  (Kernels.onlineSoftmaxKernel x y N).eraseDType

def onlineNormalizerKernel (x y : RegionName) (N : Nat) : ComputeKernel :=
  .mk [x] [] ((onlineSoftmaxKernel x y N).surfaceBody.take 4)

private theorem project_erasure (ck : ComputeKernel) (alg : Kernel)
    (h : ck.toAlgorithm? = .ok alg) :
    ck.eraseDType.toAlgorithm? = .ok ck.eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

theorem stable_projectable (x y : RegionName) (B : Nat) :
    (stableSoftmaxKernel x y B).toAlgorithm? = .ok (stableSoftmaxKernel x y B).toAlgKernel :=
  project_erasure (Kernels.stableSoftmaxKernel x y B) _ rfl

theorem online_projectable (x y : RegionName) (N : Nat) :
    (onlineSoftmaxKernel x y N).toAlgorithm? = .ok (onlineSoftmaxKernel x y N).toAlgKernel :=
  project_erasure (Kernels.onlineSoftmaxKernel x y N) _ rfl

theorem stable_projection (x y : RegionName) (B : Nat) :
    (Kernels.stableSoftmaxKernel x y B).eraseDType = stableSoftmaxKernel x y B := rfl

theorem online_projection (x y : RegionName) (N : Nat) :
    (Kernels.onlineSoftmaxKernel x y N).eraseDType = onlineSoftmaxKernel x y N := rfl

#axiomsClean stable_projection
#axiomsClean online_projection
end VeriTile.Bench.Examples.OnlineSoftmax.RealSource
