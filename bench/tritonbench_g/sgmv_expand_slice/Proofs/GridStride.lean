import bench.tritonbench_g.sgmv_expand_slice.Proofs.GridGeometry

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.GridStride
open VeriTile.Triton

/-- Ordinary two-dimensional contiguous layout. Empty tensors and ignored
singleton strides are unrestricted; only strides used by live coordinates are
fixed by contiguity. -/
structure Contiguous (rows columns rowStride columnStride : Nat) : Prop where
  row : 1 < rows → 0 < columns → rowStride = columns
  column : 0 < rows → 1 < columns → columnStride = 1

theorem coordinate_linear {rows columns rowStride columnStride r c : Nat}
    (layout : Contiguous rows columns rowStride columnStride)
    (hr : r < rows) (hc : c < columns) :
    r*rowStride+c*columnStride = r*columns+c := by
  by_cases hrows : 1 < rows
  · rw [layout.row hrows (by omega)]
    by_cases hcols : 1 < columns
    · rw [layout.column (by omega) hcols]; simp
    · have hz : c = 0 := by omega
      simp [hz]
  · have hz : r = 0 := by omega
    subst r
    by_cases hcols : 1 < columns
    · rw [layout.column (by omega) hcols]; simp
    · have hz : c = 0 := by omega
      simp [hz]

theorem clamped_offset (p : Shape) (segment : Segment) (pid rowStride columnStride : Nat)
    (layout : Contiguous p.outputRows p.outputColumns rowStride columnStride)
    (fits : SegmentFits p segment) (live : segment.active)
    (capacity : p.slice+p.columns ≤ p.outputColumns) (columns : 0 < p.columns)
    (i : TileIndex [32,32]) :
    safeRow segment (GridGeometry.localRow p pid i)*rowStride +
      safeColumn p (GridGeometry.localColumn p pid i)*columnStride =
        GridGeometry.offset p segment pid i :=
  coordinate_linear layout (safeRow_lt p segment fits live _)
    (safeColumn_lt p capacity columns _)

theorem active_injective (p : Shape) (segment : Segment) (pid rowStride columnStride : Nat)
    (layout : Contiguous p.outputRows p.outputColumns rowStride columnStride)
    (fits : SegmentFits p segment) (live : segment.active)
    (capacity : p.slice+p.columns ≤ p.outputColumns) (columns : 0 < p.columns)
    (i j : TileIndex [32,32]) (hi : GridGeometry.active p segment pid i)
    (hj : GridGeometry.active p segment pid j)
    (same : safeRow segment (GridGeometry.localRow p pid i)*rowStride +
        safeColumn p (GridGeometry.localColumn p pid i)*columnStride =
      safeRow segment (GridGeometry.localRow p pid j)*rowStride +
        safeColumn p (GridGeometry.localColumn p pid j)*columnStride) : i = j := by
  rw [clamped_offset p segment pid rowStride columnStride layout fits live capacity columns,
    clamped_offset p segment pid rowStride columnStride layout fits live capacity columns] at same
  exact GridGeometry.active_offset_injective p segment pid capacity i j hi hj same

#axiomsClean coordinate_linear
#axiomsClean clamped_offset
#axiomsClean active_injective
end VeriTile.Bench.Repairs.SgmvExpandSlice.GridStride
