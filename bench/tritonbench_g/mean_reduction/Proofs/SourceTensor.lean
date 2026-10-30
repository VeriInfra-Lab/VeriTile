import bench.tritonbench_g.mean_reduction.Proofs.WholeShape

namespace VeriTile.Bench.TritonBenchG.MeanReduction.Whole
noncomputable section
/-- FP32 element addresses are word addresses. The finite-address adapter below
maps them to aligned byte pointers by multiplying by four. Strides may be zero
or non-unit, and the input need not be contiguous. -/
structure Tensor (rank : Nat) where
  shape : Fin rank → Nat
  stride : Fin rank → Nat
  base : Nat
  device : Nat

def Tensor.address (x : Tensor rank) (i : Coordinates x.shape) : Nat :=
  x.base + ∑ ax, x.stride ax * (i ax).val

def Tensor.read (x : Tensor rank) (mem : Nat → α) : Coordinates x.shape → α :=
  fun i => mem (x.address i)


end
end VeriTile.Bench.TritonBenchG.MeanReduction.Whole
