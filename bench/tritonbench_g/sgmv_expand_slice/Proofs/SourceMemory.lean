import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceInit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceMemory
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

/-- Entry snapshot interpretation at valid coordinates only. Caller aliases
are handled when constructing these private regions, not by this predicate. -/
structure Inputs {α : Type} [Inhabited α] (abi : SourceABI) (p : KernelArgs)
    (shape : Shape) (s : State α) (X : Nat → Nat → α) (W : Nat → Nat → Nat → α) : Prop where
  inner : p.inner = shape.inner
  columns : p.columns = shape.columns
  x : ∀ r k, r < shape.tokens → k < shape.inner →
    toFloat abi.input.dtype ((s.mem p.input (r*p.inputRowStride+k*p.inputInnerStride)).read
      abi.input.dtype.toTileDType) = X r k
  weight : ∀ a c k, a < shape.adapters → c < shape.columns → k < shape.inner →
    toFloat abi.weight.scalar.dtype
      ((s.mem p.weight (p.adapterStride*a+k*p.weightInnerStride+c*p.weightColumnStride)).read
        abi.weight.scalar.dtype.toTileDType) = W a c k

theorem exact_mask (inner block : Nat) (k : Fin 16) (hb : block < (inner+15)/16) :
    (inner % 16 = 0 ∨ block*16+k.val < inner) ↔ block*16+k.val < inner := by
  have hk := k.isLt
  omega

theorem a_values {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (shape : Shape) (s : State α) (X : Nat → Nat → α)
    (W : Nat → Nat → Nat → α) (h : Inputs abi p shape s X W)
    (c : SourceBody.Coordinates) (rows : ∀ r, c.start+c.rows r < shape.tokens)
    (block : Nat) (hb : block < (p.inner+15)/16) (i : TileIndex [32,16]) :
    SourceBody.aValues M abi p c s block i =
      let value := if block*16+i.2.1.val < shape.inner then X (c.start+c.rows i.1) (block*16+i.2.1.val)
        else M.cast none .real abi.input.dtype (M.literal none .real 0)
      if abi.input = abi.weight.scalar then value
        else M.cast none abi.input.dtype abi.weight.scalar.dtype value := by
  unfold SourceBody.aValues
  simp only [SourceBody.globalInner]
  simp only [exact_mask p.inner block i.2.1 hb]
  simp only [h.inner]
  by_cases hk : block*16+i.2.1.val < shape.inner
  · have ho : SourceBody.aOffset p c block i =
        (c.start+c.rows i.1)*p.inputRowStride+(block*16+i.2.1.val)*p.inputInnerStride := by
      simp [SourceBody.aOffset,SourceBody.safeInner,SourceBody.globalInner,h.inner,hk,Nat.add_mul]
    simp only [if_pos hk,ho,h.x _ _ (rows _) hk]
  · simp only [if_neg hk]

theorem b_values {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (shape : Shape) (s : State α) (X : Nat → Nat → α)
    (W : Nat → Nat → Nat → α) (h : Inputs abi p shape s X W)
    (c : SourceBody.Coordinates) (adapter : c.adapter < shape.adapters)
    (columns : ∀ j, c.columns j < shape.columns)
    (block : Nat) (hb : block < (p.inner+15)/16) (i : TileIndex [16,32]) :
    SourceBody.bValues M abi p c s block i =
      if block*16+i.1.val < shape.inner then W c.adapter (c.columns i.2.1) (block*16+i.1.val)
        else M.cast none .real abi.weight.scalar.dtype (M.literal none .real 0) := by
  unfold SourceBody.bValues
  simp only [SourceBody.globalInner]
  simp only [exact_mask p.inner block i.1 hb]
  simp only [h.inner]
  by_cases hk : block*16+i.1.val < shape.inner
  · have ho : SourceBody.bOffset p c block i =
        p.adapterStride*c.adapter+(block*16+i.1.val)*p.weightInnerStride+c.columns i.2.1*p.weightColumnStride := by
      simp [SourceBody.bOffset,SourceBody.safeInner,SourceBody.globalInner,h.inner,hk]
    simp only [if_pos hk,ho,h.weight _ _ _ adapter (columns _) hk]
  · simp only [if_neg hk]

theorem coordinates_rows (p : KernelArgs) (shape : Shape) (segment : Segment)
    (fits : SegmentFits shape segment) (live : segment.active) (pid : Nat) (i : Fin 32) :
    segment.start+(SourceInit.coordinates p segment.start (segment.adapter.getD 0)
      segment.length pid).rows i < shape.tokens := by
  exact gathered_row_in_bounds shape segment fits live _

theorem coordinates_columns (p : KernelArgs) (shape : Shape) (segment : Segment)
    (columns : p.columns = shape.columns) (positive : 0 < shape.columns) (pid : Nat) (i : Fin 32) :
    (SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length pid).columns i < shape.columns := by
  simp only [SourceInit.coordinates,columns]
  exact Nat.mod_lt _ positive

theorem coordinates_adapter (shape : Shape) (segment : Segment)
    (fits : SegmentFits shape segment) (live : segment.active) : segment.adapter.getD 0 < shape.adapters := by
  cases ha : segment.adapter with
  | none => simp [Segment.active,ha] at live
  | some a => simpa [ha] using fits.adapter a ha live.1

#axiomsClean exact_mask
#axiomsClean a_values
#axiomsClean b_values
#axiomsClean coordinates_rows
#axiomsClean coordinates_columns
#axiomsClean coordinates_adapter
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceMemory
