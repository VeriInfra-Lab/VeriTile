import bench.repairs.tritonbench_g.sin_computation.Kernels
import VeriTile.Meta.StatementAudit

/-! Previous full programs are proof-only snapshots; current Kernels.lean
remains authoritative. Runtime sizes keep their source ABI in the signature
and are specialized to the invocation value, not reclassified as constexpr. -/
namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.SourcePreservation
open VeriTile.Triton
noncomputable section

def previous_blockSize : Nat := 4
def previous_gridCount (n : Nat) : Nat := (n + 3) / 4

def previous_kernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
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

def previous_indexCommutedKernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
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

def previous_originalKernel (n : Nat) : ComputeKernel := previous_kernel "in_ptr0" "out_ptr" n previous_blockSize
def previous_candidateKernel (n : Nat) : ComputeKernel := previous_indexCommutedKernel "in_ptr0" "out_ptr" n previous_blockSize

theorem kernel_preserved (X O : RegionName) (n B : Nat) :
    kernel X O n B = previous_kernel X O n B := rfl

theorem indexCommutedKernel_preserved (X O : RegionName) (n B : Nat) :
    indexCommutedKernel X O n B = previous_indexCommutedKernel X O n B := rfl

#axiomsClean kernel_preserved
#axiomsClean indexCommutedKernel_preserved
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.SourcePreservation
