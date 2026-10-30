import VeriTile.Triton.DSL

/-! Canonical readable FP32 sources. The retained candidate commutes only the
integer block product; every floating operation, mask and memory access stays. -/
namespace VeriTile.Bench.Repairs.SinKernel
open VeriTile.Triton

def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n + 1023) / 1024

def kernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let output_ptr : Region .fp32 := O.cast
  triton {
    pid = tl.program_id(0)
    block_start = pid * $(B)
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(n)
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.math.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def indexCommutedKernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let output_ptr : Region .fp32 := O.cast
  triton {
    pid = tl.program_id(0)
    block_start = $(B) * pid
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(n)
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.math.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def originalKernel (n : Nat) : ComputeKernel := kernel "x_ptr" "output_ptr" n blockSize
def candidateKernel (n : Nat) : ComputeKernel := indexCommutedKernel "x_ptr" "output_ptr" n blockSize
end VeriTile.Bench.Repairs.SinKernel
