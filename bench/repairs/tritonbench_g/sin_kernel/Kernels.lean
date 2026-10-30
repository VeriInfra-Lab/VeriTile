import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! Canonical readable FP32 sources. The retained candidate commutes only the
integer block product; every floating operation, mask and memory access stays. -/
namespace VeriTile.Bench.Repairs.SinKernel
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("output_ptr",.fp32)]
  scalars := [("n_elements",.i32)]
  constexpr := ["BLOCK_SIZE"]


def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n + 1023) / 1024

def kernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let x_ptr : RegionName := X
  let output_ptr : RegionName := O
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [n_elements := n] in
  triton {
    # x_ptr: pointer to input data
    # output_ptr: pointer to output data
    # n_elements: number of elements to process
    # BLOCK_SIZE: block size for Triton kernel
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.math.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def indexCommutedKernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let x_ptr : RegionName := X
  let output_ptr : RegionName := O
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [n_elements := n] in
  triton {
    # x_ptr: pointer to input data
    # output_ptr: pointer to output data
    # n_elements: number of elements to process
    # BLOCK_SIZE: block size for Triton kernel
    pid = tl.program_id(0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.math.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def originalKernel (n : Nat) : ComputeKernel := kernel "x_ptr" "output_ptr" n blockSize
def candidateKernel (n : Nat) : ComputeKernel := indexCommutedKernel "x_ptr" "output_ptr" n blockSize
end VeriTile.Bench.Repairs.SinKernel
