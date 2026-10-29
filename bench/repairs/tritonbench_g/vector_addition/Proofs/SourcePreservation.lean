import bench.repairs.tritonbench_g.vector_addition.Kernels
import VeriTile.Meta.StatementAudit

/-! Previous full programs are proof-only snapshots; current Kernels.lean
remains authoritative. Runtime sizes keep their source ABI in the signature
and are specialized to the invocation value, not reclassified as constexpr. -/
namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous.SourcePreservation
open VeriTile.Triton
noncomputable section

def previous_blockSize : Nat := 1024
def previous_gridCount (n : Nat) : Nat := (n + 1023) / 1024

/-- Authoritative original source with explicit FP32 memory and arithmetic tags. -/
def previous_add_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let y_ptr : Region .fp32 := Y.cast
  let output_ptr : Region .fp32 := O.cast
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * $(BLOCK_SIZE)
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
  }

/-- Existing ADD-COMMUTE candidate, retaining the original statement placement. -/
def previous_add_commuted_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let y_ptr : Region .fp32 := Y.cast
  let output_ptr : Region .fp32 := O.cast
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * $(BLOCK_SIZE)
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = y + x
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def previous_originalKernel (n : Nat) : ComputeKernel := previous_add_kernel "x" "y" "out" n previous_blockSize

def previous_commutedKernel (n : Nat) : ComputeKernel := previous_add_commuted_kernel "x" "y" "out" n previous_blockSize


theorem add_kernel_preserved (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) :
    add_kernel X Y O n_elements BLOCK_SIZE = previous_add_kernel X Y O n_elements BLOCK_SIZE := rfl

theorem add_commuted_kernel_preserved (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) :
    add_commuted_kernel X Y O n_elements BLOCK_SIZE = previous_add_commuted_kernel X Y O n_elements BLOCK_SIZE := rfl

#axiomsClean add_kernel_preserved
#axiomsClean add_commuted_kernel_preserved
end
end VeriTile.Bench.Repairs.VectorAddition.Contiguous.SourcePreservation
