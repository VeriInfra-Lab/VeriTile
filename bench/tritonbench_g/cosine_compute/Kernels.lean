import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! Complete FP32 cosine source and its existing integer-index variant. The
input FP32 conversion, cosine, masked load and store are shared unchanged. -/
namespace VeriTile.Bench.Repairs.CosineCompute
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("a",.fp32),("b",.fp32)]
  scalars := [("n_elements",.i32)]
  constexpr := ["BLOCK_SIZE"]


def blockSize : Nat := 256
def gridCount (n : Nat) : Nat := (n + 255) / 256

def kernel (A O : RegionName) (n B : Nat) : ComputeKernel :=
  let a : RegionName := A
  let b : RegionName := O
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [n_elements := n] in
  triton {
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos(a_value.to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)
  }

def indexCommutedKernel (A O : RegionName) (n B : Nat) : ComputeKernel :=
  let a : RegionName := A
  let b : RegionName := O
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [n_elements := n] in
  triton {
    offset = BLOCK_SIZE * tl.program_id(0) + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos(a_value.to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)
  }

def originalKernel (n : Nat) : ComputeKernel := kernel "a" "b" n blockSize
def candidateKernel (n : Nat) : ComputeKernel := indexCommutedKernel "a" "b" n blockSize
end VeriTile.Bench.Repairs.CosineCompute
