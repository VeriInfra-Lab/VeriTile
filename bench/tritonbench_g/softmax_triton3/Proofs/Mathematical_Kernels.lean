import bench.tritonbench_g.softmax_triton3.Kernels

/-! The mathematical proof uses the selected source directly. Storage labels
below belong to the existing mathematical model; there is no second quotation. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton3
open VeriTile.Triton

abbrev softmax_kernel := VeriTile.Bench.Repairs.SoftmaxTriton3.softmax_kernel

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

abbrev Source := RegionName → RegionName → RegionName → Nat → Nat → Nat → Bool → ComputeKernel

end VeriTile.Bench.TritonBenchG.SoftmaxTriton3
