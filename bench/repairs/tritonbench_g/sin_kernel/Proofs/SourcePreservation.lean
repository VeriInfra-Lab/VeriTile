import bench.repairs.tritonbench_g.sin_kernel.Kernels
import VeriTile.Meta.StatementAudit

/-! Previous full programs are proof-only snapshots; current Kernels.lean
remains authoritative. Runtime sizes keep their source ABI in the signature
and are specialized to the invocation value, not reclassified as constexpr. -/
namespace VeriTile.Bench.Repairs.SinKernel.SourcePreservation
open VeriTile.Triton
noncomputable section

def previous_blockSize : Nat := 1024
def previous_gridCount (n : Nat) : Nat := (n + 1023) / 1024

def previous_kernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
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

def previous_indexCommutedKernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
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

def previous_originalKernel (n : Nat) : ComputeKernel := previous_kernel "x_ptr" "output_ptr" n previous_blockSize
def previous_candidateKernel (n : Nat) : ComputeKernel := previous_indexCommutedKernel "x_ptr" "output_ptr" n previous_blockSize

theorem kernel_preserved (X O : RegionName) (n B : Nat) :
    kernel X O n B = previous_kernel X O n B := rfl

theorem indexCommutedKernel_preserved (X O : RegionName) (n B : Nat) :
    indexCommutedKernel X O n B = previous_indexCommutedKernel X O n B := rfl

#axiomsClean kernel_preserved
#axiomsClean indexCommutedKernel_preserved
end
end VeriTile.Bench.Repairs.SinKernel.SourcePreservation
