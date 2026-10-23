import VeriTile.Triton.DSL

/-! Canonical readable FP32 sources. The retained candidate commutes only the
integer block product; every floating operation, mask and memory access stays. -/
namespace VeriTile.Bench.Repairs.SinComputation.Contiguous
open VeriTile.Triton

def blockSize : Nat := 4
def gridCount (n : Nat) : Nat := (n + 3) / 4

def kernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let in_ptr0 : Region .fp32 := X.cast
  let out_ptr : Region .fp32 := O.cast
  triton {
    pid = tl.program_id(0)
    block_start = pid * $(B)
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(n)
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)
  }

def indexCommutedKernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let in_ptr0 : Region .fp32 := X.cast
  let out_ptr : Region .fp32 := O.cast
  triton {
    pid = tl.program_id(0)
    block_start = $(B) * pid
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(n)
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)
  }

def originalKernel (n : Nat) : ComputeKernel := kernel "in_ptr0" "out_ptr" n blockSize
def candidateKernel (n : Nat) : ComputeKernel := indexCommutedKernel "in_ptr0" "out_ptr" n blockSize
end VeriTile.Bench.Repairs.SinComputation.Contiguous
