import bench.tritonbench_g.rmsnorm_triton.Kernels

/-! Kernel geometry only; runtime Tensor metadata and memory execution are
kept in their existing application/proof files. -/
namespace VeriTile.Bench.Repairs.RmsnormTriton.KernelGrid
open VeriTile.Triton
noncomputable section

/-- Only kernel address data; no Tensor object, device, dtype flag or API law. -/
structure Buffer where
  allocation : Nat
  base : Nat
  stride : Nat → Nat

structure Geometry where
  batches : Nat
  rows : Nat
  width : Nat
  input : Buffer
  weights : Buffer
  output : Buffer
  eps : ℝ

def Geometry.parameters (g : Geometry) : Parameters :=
  ⟨g.batches,g.rows,g.width,
    ⟨[g.batches,g.rows,g.width],g.input.stride,g.input.allocation,g.input.base,0,.fp32⟩,
    ⟨[g.width],g.weights.stride,g.weights.allocation,g.weights.base,0,.fp32⟩,
    ⟨[g.batches,g.rows,g.width],g.output.stride,g.output.allocation,g.output.base,0,.fp32⟩,g.eps⟩


end
end VeriTile.Bench.Repairs.RmsnormTriton.KernelGrid
