import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Geometry
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Snapshot

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Snapshot
open VeriTile.Triton Geometry
abbrev Logical := VeriTile.Bench.Repairs.BgmvExpandSlice.Snapshot.Logical

/-- All four logical snapshots precede the sole caller write. These ordinary
coordinate-copy contracts permit all cross-role aliases, offsets, strides,
broadcast reads, rank-4 singleton squeeze and lazy negative views. -/
structure Packed (α : Type) (h : Call) (logical : Logical α) (zero : α) where
  input : Nat → α
  weights : Nat → α
  output : Nat → α
  indices : Nat → Int
  inputCopy : ∀ b k,b<h.batches → k<h.K → input (inputOffset h b k)=logical.input b k
  inputZero : ∀ b k,b<h.batches → h.K≤k → k<h.paddedK → input (inputOffset h b k)=zero
  weightCopy : ∀ a n k,a<h.adapters → n<h.N → k<h.K →
    weights (weightOffset h a n k)=logical.weight a n k
  weightZero : ∀ a n k,a<h.adapters → n<h.BN → k<h.paddedK → (h.N≤n ∨ h.K≤k) →
    weights (weightOffset h a n k)=zero
  outputCopy : ∀ b n,b<h.batches → n<h.N → output (outputOffset h b n)=logical.output b n
  metadataCopy : ∀ b,b<h.batches → indices b=logical.indices b

end VeriTile.Bench.Repairs.BgmvShrinkKernel.Snapshot
