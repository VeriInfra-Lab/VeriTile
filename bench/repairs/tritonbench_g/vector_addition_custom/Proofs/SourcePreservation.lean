import bench.repairs.tritonbench_g.vector_addition_custom.Kernels
import VeriTile.Meta.StatementAudit

/-! Previous full programs are proof-only snapshots; current Kernels.lean
remains authoritative. Runtime sizes keep their source ABI in the signature
and are specialized to the invocation value, not reclassified as constexpr. -/
namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.SourcePreservation
open VeriTile.Triton
noncomputable section

def previous_blockSize : Nat := 16
def previous_gridCount (n : Nat) : Nat := (n + 15) / 16

/-- Authoritative original source with explicit FP32 memory and arithmetic tags. -/
def previous_add_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let y_ptr : Region .fp32 := Y.cast
  let output_ptr : Region .fp32 := O.cast
  triton {
    prog_id = tl.program_id(0)
    offs = prog_id * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(x_ptr + offs, mask=offs < $(n_elements))
    b = tl.load(y_ptr + offs, mask=offs < $(n_elements))
    tl.store(output_ptr + offs, a + b, mask=offs < $(n_elements))
  }

/-- Existing ADD-COMMUTE candidate, retaining the original statement placement. -/
def previous_add_commuted_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let y_ptr : Region .fp32 := Y.cast
  let output_ptr : Region .fp32 := O.cast
  triton {
    prog_id = tl.program_id(0)
    offs = prog_id * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(x_ptr + offs, mask=offs < $(n_elements))
    b = tl.load(y_ptr + offs, mask=offs < $(n_elements))
    tl.store(output_ptr + offs, b + a, mask=offs < $(n_elements))
  }

def previous_originalKernel (n : Nat) : ComputeKernel := previous_add_kernel "x" "y" "out" n previous_blockSize

def previous_commutedKernel (n : Nat) : ComputeKernel := previous_add_commuted_kernel "x" "y" "out" n previous_blockSize

/-- Historical finite-interpreter name; this is the same canonical DSL source. -/
abbrev previous_fp32Kernel (X Y O : RegionName) (n : Nat) : ComputeKernel :=
  previous_add_kernel X Y O n previous_blockSize


theorem add_kernel_preserved (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) :
    add_kernel X Y O n_elements BLOCK_SIZE = previous_add_kernel X Y O n_elements BLOCK_SIZE := rfl

theorem add_commuted_kernel_preserved (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) :
    add_commuted_kernel X Y O n_elements BLOCK_SIZE = previous_add_commuted_kernel X Y O n_elements BLOCK_SIZE := rfl

#axiomsClean add_kernel_preserved
#axiomsClean add_commuted_kernel_preserved
end
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.SourcePreservation
