import VeriTile.Triton
import bench.tritonbench_g.masked_select.Proofs.Sources.Mathematical_Kernels

/-! Independent coordinate helpers for the selected masked scatter. The
canonical executable program is in Kernels.lean; the complete signed-prefix
execution and stable-selection proofs are in Mathematical_TypedCopy,
Mathematical_WholeExecution and PublicIO. Inactive lanes do not access the
output even when the source's signed prefix-minus-one value is negative. -/

namespace VeriTile.Bench.TritonBenchG.MaskedSelect

open VeriTile.Triton
open scoped VeriTile.Triton.BoolScatterMasked2DKernelIO₁

/-- Lane `i`'s address in all three read windows of program `s.pid`. Internal
to the readback lemmas (the `⊨` signature declares the same window as
`fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val`). -/
def maskedSelectOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

/-- Lane `i`'s state-coupled scatter destination: the masked `prefix_sum` load
minus one (the kernel's `out_offset`). Internal to the readback lemmas; the `⊨`
signature re-expresses it as `ids j − 1` in terms of the pinned index tile. -/
def maskedSelectStoreOffset
    (s : BlockState) (prefix_sum_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  ((if maskedSelectOffset s BLOCK_SIZE i < n_elements then
      s.readMemValue .int prefix_sum_ptr (maskedSelectOffset s BLOCK_SIZE i)
    else
      0) - 1).toNat

/-- Lane `i` is **write-active**: in bounds and selected — the state-coupled
form of the signature's `writeMask`. -/
def active
    (s : BlockState) (select_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  maskedSelectOffset s BLOCK_SIZE i < n_elements ∧
    s.readMemValue .bool select_mask_ptr (maskedSelectOffset s BLOCK_SIZE i) = Bool.true

instance activeDecidable
    (s : BlockState) (select_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active s select_mask_ptr n_elements BLOCK_SIZE i) := by
  unfold active
  infer_instance

end VeriTile.Bench.TritonBenchG.MaskedSelect
