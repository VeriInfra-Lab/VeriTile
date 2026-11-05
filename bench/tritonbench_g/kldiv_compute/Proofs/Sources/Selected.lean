import bench.tritonbench_g.kldiv_compute.Kernels

/-! The selected FP32 model references the canonical raw Python bodies. -/
namespace VeriTile.Bench.Repairs.KldivCompute.SelectedSource
open VeriTile.Triton

def original (X Y O : RegionName) (n block : Nat) : ComputeKernel :=
  KernelDSL.kernel ⟨X,Y,O,n,block⟩

def commuted (X Y O : RegionName) (n block : Nat) : ComputeKernel :=
  KernelDSL.commutedKernel ⟨X,Y,O,n,block⟩

end VeriTile.Bench.Repairs.KldivCompute.SelectedSource
