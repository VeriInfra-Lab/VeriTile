import bench.repairs.tritonbench_g.add_value.Kernels
import VeriTile.Meta.StatementAudit

/-! Previous full programs are proof-only snapshots; current Kernels.lean
remains authoritative. Runtime sizes keep their source ABI in the signature
and are specialized to the invocation value, not reclassified as constexpr. -/
namespace VeriTile.Bench.Repairs.AddValue.Contiguous.SourcePreservation
open VeriTile.Triton
noncomputable section

def previous_blockSize : Nat := 1024
def previous_gridCount (n : Nat) : Nat := (n + 1023) / 1024
def previous_scalarWord : FP.Value .fp32 := ⟨0x41200000#32⟩
def previous_mathematical (xs : Nat → ℝ) (i : Nat) : ℝ := xs i + 10

/-- Checked `x + 10` source. Typed regions retain explicit FP32 load,
computation and store precision; no handwritten AST is maintained beside it. -/
def previous_kernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let output_ptr : Region .fp32 := O.cast
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * $(B)
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(n)
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + 10
    tl.store(output_ptr + offsets, output, mask=mask)
  }

/-- The actual wrapper selection of the authoritative source. -/
def previous_originalKernel (n : Nat) : ComputeKernel := previous_kernel "x" "out" n previous_blockSize

/-- ADD-COMMUTE candidate. The scalar is a local constant, not a previous_kernel input;
load, mask, addition precision and store match the original source. -/
def previous_commutedKernel (n : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let output_ptr : Region .fp32 := ⟨"out"⟩
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * $(previous_blockSize)
    offsets = block_start + tl.arange(0, $(previous_blockSize))
    mask = offsets < $(n)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = 10
    output = y + x
    tl.store(output_ptr + offsets, output, mask=mask)
  }


theorem kernel_preserved (X O : RegionName) (n B : Nat) :
    kernel X O n B = previous_kernel X O n B := rfl

theorem commutedKernel_preserved (n : Nat) :
    commutedKernel n = previous_commutedKernel n := rfl

#axiomsClean kernel_preserved
#axiomsClean commutedKernel_preserved
end
end VeriTile.Bench.Repairs.AddValue.Contiguous.SourcePreservation
