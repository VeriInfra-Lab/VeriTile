import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceExecution
import bench.tritonbench_g.sgmv_expand_slice.Proofs.GridGeometry

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.TileCoordinates
open VeriTile.Triton

theorem active_iff (p : KernelArgs) (segment : Segment) (pid : Nat) (i : TileIndex [32,32]) :
    SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length pid) i ↔
      i.1.val+pid/tiles p.columns*32 < segment.length ∧
      i.2.1.val+pid%tiles p.columns*32 < p.columns := by
  simp only [SourceSuffix.active,SourceSuffix.row,SourceSuffix.column,SourceExecution.outputCoordinates]
  omega

theorem row_div (p : KernelArgs) (segment : Segment) (pid : Nat) (i : TileIndex [32,32]) :
    (SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length pid) i-segment.start)/32 =
      pid/tiles p.columns := by
  have hi := i.1.isLt
  simp only [SourceSuffix.row,SourceExecution.outputCoordinates]
  omega

theorem row_mod (p : KernelArgs) (segment : Segment) (pid : Nat) (i : TileIndex [32,32]) :
    (SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length pid) i-segment.start)%32 =
      i.1.val := by
  have hi := i.1.isLt
  simp only [SourceSuffix.row,SourceExecution.outputCoordinates]
  omega

theorem column_div (p : KernelArgs) (segment : Segment) (pid : Nat) (i : TileIndex [32,32]) :
    (SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length pid) i-p.slice)/32 =
      pid%tiles p.columns := by
  have hi := i.2.1.isLt
  simp only [SourceSuffix.column,SourceExecution.outputCoordinates]
  omega

theorem column_mod (p : KernelArgs) (segment : Segment) (pid : Nat) (i : TileIndex [32,32]) :
    (SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length pid) i-p.slice)%32 =
      i.2.1.val := by
  have hi := i.2.1.isLt
  simp only [SourceSuffix.column,SourceExecution.outputCoordinates]
  omega

theorem active_cell (p : KernelArgs) (shape : Shape) (segment : Segment) (pid : Nat)
    (columns : p.columns = shape.columns) (slice : p.slice = shape.slice) (live : segment.active)
    (i : TileIndex [32,32])
    (hi : SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length pid) i) :
    ActiveCell shape segment
      (SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length pid) i)
      (SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length pid) i) := by
  refine ⟨live,?_,hi.1,?_,?_⟩
  · simp only [SourceSuffix.row,SourceExecution.outputCoordinates]; omega
  · simp only [SourceSuffix.column,SourceExecution.outputCoordinates]; omega
  · simpa only [slice,columns] using hi.2

theorem gathered_row (p : KernelArgs) (segment : Segment) (pid : Nat) (i : TileIndex [32,32])
    (hi : SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length pid) i) :
    segment.start+(SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length pid).rows i.1 =
      SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length pid) i := by
  have h := (active_iff p segment pid i).mp hi
  simp [SourceInit.coordinates,SourceSetup.rowIndex,SourceSuffix.row,SourceExecution.outputCoordinates,
    Nat.mod_eq_of_lt h.1,Nat.add_assoc]

theorem gathered_column (p : KernelArgs) (segment : Segment) (pid : Nat) (i : TileIndex [32,32])
    (hi : SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length pid) i) :
    (SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length pid).columns i.2.1 =
      SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length pid) i-p.slice := by
  have h := (active_iff p segment pid i).mp hi
  simp [SourceInit.coordinates,SourceSetup.columnIndex,SourceSuffix.column,SourceExecution.outputCoordinates,
    Nat.mod_eq_of_lt h.2]

#axiomsClean active_iff
#axiomsClean row_div
#axiomsClean row_mod
#axiomsClean column_div
#axiomsClean column_mod
#axiomsClean active_cell
#axiomsClean gathered_row
#axiomsClean gathered_column
end VeriTile.Bench.Repairs.SgmvExpandSlice.TileCoordinates
