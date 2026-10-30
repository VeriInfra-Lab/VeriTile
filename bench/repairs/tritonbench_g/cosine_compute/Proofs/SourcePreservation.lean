import bench.repairs.tritonbench_g.cosine_compute.Kernels
import VeriTile.Meta.StatementAudit

/-! Previous full programs are proof-only snapshots; current Kernels.lean
remains authoritative. Runtime sizes keep their source ABI in the signature
and are specialized to the invocation value, not reclassified as constexpr. -/
namespace VeriTile.Bench.Repairs.CosineCompute.SourcePreservation
open VeriTile.Triton
noncomputable section

def previous_blockSize : Nat := 256
def previous_gridCount (n : Nat) : Nat := (n + 255) / 256

def previous_kernel (A O : RegionName) (n B : Nat) : ComputeKernel :=
  let a : Region .fp32 := A.cast
  let b : Region .fp32 := O.cast
  triton {
    offset = tl.program_id(0) * $(B) + tl.arange(0, $(B))
    mask = offset < $(n)
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos((a_value).to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)
  }

def previous_indexCommutedKernel (A O : RegionName) (n B : Nat) : ComputeKernel :=
  let a : Region .fp32 := A.cast
  let b : Region .fp32 := O.cast
  triton {
    offset = $(B) * tl.program_id(0) + tl.arange(0, $(B))
    mask = offset < $(n)
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos((a_value).to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)
  }

def previous_originalKernel (n : Nat) : ComputeKernel := previous_kernel "a" "b" n previous_blockSize
def previous_candidateKernel (n : Nat) : ComputeKernel := previous_indexCommutedKernel "a" "b" n previous_blockSize

theorem kernel_preserved (A O : RegionName) (n B : Nat) :
    kernel A O n B = previous_kernel A O n B := rfl

theorem indexCommutedKernel_preserved (A O : RegionName) (n B : Nat) :
    indexCommutedKernel A O n B = previous_indexCommutedKernel A O n B := rfl

#axiomsClean kernel_preserved
#axiomsClean indexCommutedKernel_preserved
end
end VeriTile.Bench.Repairs.CosineCompute.SourcePreservation
