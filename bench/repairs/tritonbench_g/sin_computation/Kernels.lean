import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! Canonical readable FP32 sources. The retained candidate commutes only the
integer block product; every floating operation, mask and memory access stays. -/
namespace VeriTile.Bench.Repairs.SinComputation.Contiguous
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("in_ptr0",.fp32),("out_ptr",.fp32)]
  scalars := [("n_elements",.i32)]
  constexpr := ["BLOCK_SIZE"]


def blockSize : Nat := 4
def gridCount (n : Nat) : Nat := (n + 3) / 4

def kernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let in_ptr0 : RegionName := X
  let out_ptr : RegionName := O
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [n_elements := n] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)
  }

def indexCommutedKernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let in_ptr0 : RegionName := X
  let out_ptr : RegionName := O
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [n_elements := n] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)
  }

def originalKernel (n : Nat) : ComputeKernel := kernel "in_ptr0" "out_ptr" n blockSize
def candidateKernel (n : Nat) : ComputeKernel := indexCommutedKernel "in_ptr0" "out_ptr" n blockSize
end VeriTile.Bench.Repairs.SinComputation.Contiguous
