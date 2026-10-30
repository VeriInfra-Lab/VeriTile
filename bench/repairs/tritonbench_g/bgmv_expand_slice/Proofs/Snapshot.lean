import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Geometry

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Snapshot
open VeriTile.Triton

/-- Caller-entry logical values, after interpreting every source view's
storage offset, strides and lazy negative flag. Aliases are permitted: all
four operands are observed before the caller output is changed. -/
structure Logical (α : Type) where
  input : Nat → Nat → α
  weight : Nat → Nat → Nat → α
  output : Nat → Nat → α
  indices : Nat → Int

/-- Ordinary resolve_neg/zeros/copy_/clone snapshot contracts. Only local
coordinate transport and zero fill appear; no GEMV result or source success
is assumed. All fields belong to the same wrapper call. -/
structure Packed (α : Type) (h : Call) (logical : Logical α) (zero : α) where
  input : Nat → α
  weights : Nat → α
  output : Nat → α
  indices : Nat → Int
  inputCopy : ∀ b k,b<h.batches → k<h.K → input (b*h.BK+k)=logical.input b k
  inputZero : ∀ b k,b<h.batches → h.K≤k → k<h.BK → input (b*h.BK+k)=zero
  weightCopy : ∀ a n k,a<h.adapters → n<h.N → k<h.K →
    weights (a*((h.N+255)*h.BK)+n*h.BK+k)=logical.weight a n k
  weightZero : ∀ a n k,a<h.adapters → n<h.N+255 → k<h.BK → (h.N≤n ∨ h.K≤k) →
    weights (a*((h.N+255)*h.BK)+n*h.BK+k)=zero
  outputCopy : ∀ b c,b<h.batches → c<h.columns →
    output (b*(h.columns+255)+c)=logical.output b c
  metadataCopy : ∀ b,b<h.batches → indices b=logical.indices b

/-- The requested grid contains each (row,batch) once. Native launch-product
admission and any segmentation remain the shared runtime obligation. -/
structure Schedule (h : Call) where
  programs : List (Geometry.Program h)
  unique : programs.Pairwise (fun p q =>
    Geometry.outputOffset h p.batch p.row 0≠Geometry.outputOffset h q.batch q.row 0)
  covers : ∀ p : Geometry.Program h,p∈programs

end VeriTile.Bench.Repairs.BgmvExpandSlice.Snapshot
