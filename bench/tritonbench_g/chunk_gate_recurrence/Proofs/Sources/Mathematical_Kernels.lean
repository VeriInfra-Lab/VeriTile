import bench.tritonbench_g.chunk_gate_recurrence.Proofs.Sources.Selected

/-! Storage-polymorphic invocation of the same copied backward bodies. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedSource
open VeriTile.Triton
def specializeBackward (gate incoming : Storage) (swapped : Bool) (source : FloatDType) :=
  if swapped then
    VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels.backwardBodySumLeftKernel
      source gate incoming
  else
    VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels.backwardKernel
      source gate incoming

def backward (source gate incoming : Storage) :=
  specializeBackward gate incoming Bool.false source.dtype
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedSource
