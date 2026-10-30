import VeriTile.Triton.DSL

/-! Complete FP32 cosine source and its existing integer-index variant. The
input FP32 conversion, cosine, masked load and store are shared unchanged. -/
namespace VeriTile.Bench.Repairs.CosineCompute
open VeriTile.Triton

def blockSize : Nat := 256
def gridCount (n : Nat) : Nat := (n + 255) / 256

def kernel (A O : RegionName) (n B : Nat) : ComputeKernel :=
  let a : Region .fp32 := A.cast
  let b : Region .fp32 := O.cast
  triton {
    offset = tl.program_id(0) * $(B) + tl.arange(0, $(B))
    mask = offset < $(n)
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos((a_value).to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)
  }

def indexCommutedKernel (A O : RegionName) (n B : Nat) : ComputeKernel :=
  let a : Region .fp32 := A.cast
  let b : Region .fp32 := O.cast
  triton {
    offset = $(B) * tl.program_id(0) + tl.arange(0, $(B))
    mask = offset < $(n)
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos((a_value).to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)
  }

def originalKernel (n : Nat) : ComputeKernel := kernel "a" "b" n blockSize
def candidateKernel (n : Nat) : ComputeKernel := indexCommutedKernel "a" "b" n blockSize
end VeriTile.Bench.Repairs.CosineCompute
