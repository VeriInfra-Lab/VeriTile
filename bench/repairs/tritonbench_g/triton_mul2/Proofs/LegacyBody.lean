import VeriTile.Triton.DSL

/-! Canonical readable FP32 doubling kernels. Ordinary and in-place output use
the same load-before-store body; the candidate changes only multiplication
operand order. Correct and FPEquiv refer to these exact sources. -/
namespace VeriTile.Bench.Repairs.TritonMul2.LegacyBody
open VeriTile.Triton

def sourceKernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let in_ptr0 : Region .fp32 := X.cast
  let out_ptr : Region .fp32 := O.cast
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * $(B)
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(n)
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = 2 * x
    tl.store(out_ptr + offsets, output, mask=mask)
  }

/-- In-place specialization: both ports denote the same memory region. -/
def inplaceSourceKernel (X : RegionName) (n B : Nat) : ComputeKernel :=
  sourceKernel X X n B

def commutedSourceKernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let in_ptr0 : Region .fp32 := X.cast
  let out_ptr : Region .fp32 := O.cast
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * $(B)
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(n)
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = x * 2
    tl.store(out_ptr + offsets, output, mask=mask)
  }

def inplaceCommutedSourceKernel (X : RegionName) (n B : Nat) : ComputeKernel :=
  commutedSourceKernel X X n B

def grid (n block : Nat) : Nat := (n+block-1)/block
end VeriTile.Bench.Repairs.TritonMul2.LegacyBody
