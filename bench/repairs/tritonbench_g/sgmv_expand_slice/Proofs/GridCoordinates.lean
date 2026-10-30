import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.TileCoordinates
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.GridFold
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceReadback

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.GridCoordinates
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000

theorem row_eq (p : KernelArgs) (shape : Shape) (segment : Segment) (pid : Nat)
    (columns : p.columns = shape.columns) (i : TileIndex [32,32]) :
    SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length pid) i =
      GridGeometry.row shape segment pid i := by
  simp [SourceSuffix.row,SourceExecution.outputCoordinates,GridGeometry.row,GridGeometry.localRow,
    columns,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]

theorem column_eq (p : KernelArgs) (shape : Shape) (segment : Segment) (pid : Nat)
    (columns : p.columns = shape.columns) (slice : p.slice = shape.slice) (i : TileIndex [32,32]) :
    SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length pid) i =
      GridGeometry.column shape pid i := by
  simp [SourceSuffix.column,SourceExecution.outputCoordinates,GridGeometry.column,GridGeometry.localColumn,
    columns,slice,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]

theorem active_eq (p : KernelArgs) (shape : Shape) (segment : Segment) (pid : Nat)
    (columns : p.columns = shape.columns) (i : TileIndex [32,32]) :
    SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length pid) i ↔
      GridGeometry.active shape segment pid i := by
  rw [TileCoordinates.active_iff]
  simp only [GridGeometry.active,GridGeometry.localRow,GridGeometry.localColumn,columns]
  omega

theorem lane_owner (p : KernelArgs) (shape : Shape) (segment : Segment) (pid : Nat)
    (columns : p.columns = shape.columns) (slice : p.slice = shape.slice) (i : TileIndex [32,32]) :
    GridFold.owner shape segment
      (SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length pid) i)
      (SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length pid) i) = pid := by
  unfold GridFold.owner
  rw [←slice,TileCoordinates.row_div,TileCoordinates.column_div,←columns]
  simpa only [Nat.mul_comm] using Nat.div_add_mod pid (tiles p.columns)

theorem covered_iff (p : KernelArgs) (shape : Shape) (segment : Segment) (pid r c : Nat)
    (columns : p.columns = shape.columns) (slice : p.slice = shape.slice) (live : segment.active) :
    (∃ i, SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length pid) i ∧
      SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length pid) i = r ∧
      SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length pid) i = c) ↔
      ActiveCell shape segment r c ∧ GridFold.owner shape segment r c = pid := by
  constructor
  · rintro ⟨i,hi,hr,hc⟩
    exact ⟨hr ▸ hc ▸ TileCoordinates.active_cell p shape segment pid columns slice live i hi,
      hr ▸ hc ▸ lane_owner p shape segment pid columns slice i⟩
  · rintro ⟨hactive,howner⟩
    have hr : r-segment.start < segment.length := by have := hactive.2.1; have := hactive.2.2.1; omega
    have hc : c-shape.slice < shape.columns := by have := hactive.2.2.2.1; have := hactive.2.2.2.2; omega
    obtain ⟨qid,_,i,hi,hri,hci⟩ := GridGeometry.coordinate_covered shape segment (r-segment.start) (c-shape.slice) hr hc
    have hri' : SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length qid) i = r := by
      rw [row_eq p shape segment qid columns i,hri]
      exact Nat.add_sub_of_le hactive.2.1
    have hci' : SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length qid) i = c := by
      rw [column_eq p shape segment qid columns slice i,hci]
      exact Nat.add_sub_of_le hactive.2.2.2.1
    have hqid : qid = pid := by
      have h := lane_owner p shape segment qid columns slice i
      rw [hri',hci',howner] at h
      exact h.symm
    subst qid
    exact ⟨i,(active_eq p shape segment pid columns i).mpr hi,hri',hci'⟩

theorem miss_offset (p : KernelArgs) (shape : Shape) (segment : Segment) (pid r c : Nat)
    (columns : p.columns = shape.columns) (slice : p.slice = shape.slice) (live : segment.active)
    (fits : SegmentFits shape segment) (capacity : shape.slice+shape.columns ≤ shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride)
    (hr : r < shape.outputRows) (hc : c < shape.outputColumns)
    (miss : ¬(ActiveCell shape segment r c ∧ GridFold.owner shape segment r c = pid))
    (i : TileIndex [32,32])
    (hi : SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length pid) i) :
    r*p.outputRowStride+c*p.outputColumnStride ≠
      SourceSuffix.offset p (SourceExecution.outputCoordinates p segment.start segment.length pid) i := by
  intro he
  have hir := Nat.lt_of_lt_of_le hi.1 (fits.output live)
  have hic : SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length pid) i <
      shape.outputColumns := by have h := hi.2; rw [columns,slice] at h; omega
  rw [SourceSuffix.active_offset p _ i hi,GridStride.coordinate_linear layout hr hc,
    GridStride.coordinate_linear layout hir hic] at he
  obtain ⟨her,hec⟩ := linear_injective hc hic he
  exact miss ((covered_iff p shape segment pid r c columns slice live).mp ⟨i,hi,her.symm,hec.symm⟩)

#axiomsClean row_eq
#axiomsClean column_eq
#axiomsClean active_eq
#axiomsClean lane_owner
#axiomsClean covered_iff
#axiomsClean miss_offset
end VeriTile.Bench.Repairs.SgmvExpandSlice.GridCoordinates
