import bench.tritonbench_g.diag_ssm_triton.Proofs.Support.KernelEntryAdapters

/-! Typed invocation views of the same original Python bodies. The existing
fp32/fp16/bf16 mathematical API is retained; complex64 always uses fp32
components. The selected repaired host accepts float32/complex64. -/
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource
open VeriTile.Triton

abbrev forwardSource := RegionName → RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel
def forward (storage : Storage) (swapped : Bool) : forwardSource :=
  if swapped then VeriTile.Bench.Repairs.DiagSsmTriton.forwardCandidateKernel storage
  else VeriTile.Bench.Repairs.DiagSsmTriton.forwardKernel storage

abbrev backwardSource := RegionName → RegionName → RegionName → RegionName → RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel
def backward (storage : Storage) (swapped : Bool) : backwardSource :=
  if swapped then VeriTile.Bench.Repairs.DiagSsmTriton.backwardCandidateKernel storage
  else VeriTile.Bench.Repairs.DiagSsmTriton.backwardKernel storage

abbrev complexForwardSource := RegionName → RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel
def complexForward (swapped : Bool) : complexForwardSource :=
  if swapped then VeriTile.Bench.Repairs.DiagSsmTriton.complexForwardCandidateKernel
  else VeriTile.Bench.Repairs.DiagSsmTriton.complexForwardKernel

abbrev complexBackwardSource := RegionName → RegionName → RegionName → RegionName → RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel
def complexBackward (swapped : Bool) : complexBackwardSource :=
  if swapped then VeriTile.Bench.Repairs.DiagSsmTriton.complexBackwardCandidateKernel
  else VeriTile.Bench.Repairs.DiagSsmTriton.complexBackwardKernel

end VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton
open VeriTile.Triton

/-- Buffer names and logical sizes shared by the real and complex forward
specifications. B is the source block size; length is supplied separately. -/
structure ForwardLayout where
  s : RegionName
  x : RegionName
  y : RegionName
  l : RegionName
  batch : Nat
  dim : Nat
  B : Nat

/-- Backward buffers contain one initial state/transition per column,
time-dependent y/grad_y/grad_x and terminal grad_s/grad_lambda. -/
structure BackwardLayout where
  s : RegionName
  l : RegionName
  y : RegionName
  gs : RegionName
  gx : RegionName
  gl : RegionName
  gy : RegionName
  batch : Nat
  dim : Nat
  B : Nat
end VeriTile.Bench.TritonBenchG.DiagSsmTriton
