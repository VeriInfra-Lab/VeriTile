import bench.tritonbench_g.decay_cumsum.Proofs.Sources.Selected

/-! Typed invocation profiles of the same pasted source, including the
pre-existing fp16/bf16 mathematical APIs. No whole-kernel generator is used. -/
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
open VeriTile.Triton
noncomputable section

def forward (storage : Storage) (swapped : Bool) :=
  if swapped then VeriTile.Bench.Repairs.DecayCumsum.forwardCandidateKernel storage
  else VeriTile.Bench.Repairs.DecayCumsum.forwardKernel storage

def prepare (source gate : Storage) (swapped : Bool) :=
  if swapped then VeriTile.Bench.Repairs.DecayCumsum.prepareCandidateKernel source gate
  else VeriTile.Bench.Repairs.DecayCumsum.prepareKernel source gate

def backward (source gate : Storage) (swapped : Bool) :=
  if swapped then VeriTile.Bench.Repairs.DecayCumsum.backwardCandidateKernel source gate
  else VeriTile.Bench.Repairs.DecayCumsum.backwardKernel source gate

end
end VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
