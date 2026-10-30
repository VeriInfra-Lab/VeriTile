import bench.repairs.tritonbench_g.dropout_triton.Kernels

/-! Historical proof skeleton with an extra identity store cast. It is not
the current Python authority. Source proves full operational connections under
the existing identity-cast backend and the complete Real interpretation. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.LegacyPastedSurface
open VeriTile.Triton

def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n + 1023) / 1024

def sourceKernel (X K O : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  triton {
    tl.region X = tl.float32, K = tl.int32, O = tl.float32
    pid = tl.program_id(axis=0)
    block_start = pid * $(B)
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(n)
    x = tl.load(X + offsets, mask=mask)
    x_keep = tl.load(K + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(O + offsets, (output).to(tl.float32), mask=mask)
  }

def commutedSourceKernel (X K O : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  triton {
    tl.region X = tl.float32, K = tl.int32, O = tl.float32
    pid = tl.program_id(axis=0)
    block_start = $(B) * pid
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(n)
    x = tl.load(X + offsets, mask=mask)
    x_keep = tl.load(K + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(O + offsets, (output).to(tl.float32), mask=mask)
  }

def kernel (n : Nat) (p : ℝ) : ComputeKernel :=
  sourceKernel "x" "keep" "out" n blockSize p

def indexCommutedKernel (n : Nat) (p : ℝ) : ComputeKernel :=
  commutedSourceKernel "x" "keep" "out" n blockSize p

end VeriTile.Bench.Repairs.DropoutTriton.LegacyPastedSurface
