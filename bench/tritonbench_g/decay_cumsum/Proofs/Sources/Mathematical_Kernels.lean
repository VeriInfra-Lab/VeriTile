import bench.tritonbench_g.decay_cumsum.Proofs.Sources.Selected

/-! Typed invocation profiles of the same pasted source, including the
pre-existing fp16/bf16 mathematical APIs. No whole-kernel generator is used. -/
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
open VeriTile.Triton
noncomputable section

def forward (storage : Storage) (swapped : Bool) :=
  if swapped then VeriTile.Bench.Repairs.DecayCumsum.Bodies.forwardCandidateBody storage
  else VeriTile.Bench.Repairs.DecayCumsum.Bodies.forwardBody storage

def prepare (source gate : Storage) (swapped : Bool) :=
  if swapped then VeriTile.Bench.Repairs.DecayCumsum.Bodies.prepareCandidateBody source gate
  else VeriTile.Bench.Repairs.DecayCumsum.Bodies.prepareBody source gate

def backward (source gate : Storage) (swapped : Bool) :=
  if swapped then VeriTile.Bench.Repairs.DecayCumsum.Bodies.backwardCandidateBody source gate
  else VeriTile.Bench.Repairs.DecayCumsum.Bodies.backwardBody source gate

end
end VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
