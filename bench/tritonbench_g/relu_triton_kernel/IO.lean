import bench.tritonbench_g.relu_triton_kernel.Kernels
import VeriTile.Triton.Float.Kernel

/-! Public FP IO for the existing repaired ReLU store-inline pair. Both sources
share the same windows, FP32 profile, comparison port and unrestricted domain. -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel.FloatingIO
open VeriTile Triton
noncomputable section

def originalIO (N : Nat) : KernelIO₁ where
  kernel := originalKernel N
  inp := "x"
  out := "out"
  Bin := blockSize
  Bout := blockSize
  read := fun pid => pid * blockSize
  write := fun pid => pid * blockSize

def inlinedIO (N : Nat) : KernelIO₁ :=
  (originalIO N).withKernel (inlinedKernel N)

def original (N : Nat) : FP.Kernel₁ where
  io := originalIO N
  profile := FP.fp32
  domain := fun _ => .top
  comparisons := [.le .fp32 .real]

def inlined (N : Nat) : FP.Kernel₁ :=
  { original N with io := inlinedIO N }

def assumptions : Spec.Assumptions FP.GuardedFragment := []

end
end VeriTile.Bench.Repairs.ReluTritonKernel.FloatingIO

namespace VeriTile.Bench.Repairs.ReluTritonKernel.RealIO
open VeriTile.Triton
noncomputable section

/-- Every program owns its full active input/output tile. -/
def io (N : Nat) : MaskedKernelIO₁ where
  kernel := originalKernel N
  projection := by rfl
  inp := "x"
  out := "out"
  B := blockSize
  read := fun pid => pid * blockSize
  write := fun pid => pid * blockSize
  mask := fun pid i => pid * blockSize + i.val < N

end
end VeriTile.Bench.Repairs.ReluTritonKernel.RealIO
