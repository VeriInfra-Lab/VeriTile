import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceExecution
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceReadback

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceTile
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000

theorem pid_guard (columns length pid : Nat) (hn : 0 < columns)
    (hp : pid < tiles length*tiles columns) : pid / tiles columns*32 < length := by
  have ht := tiles_positive hn
  have hd : pid / tiles columns < tiles length := (Nat.div_lt_iff_lt_mul ht).mpr hp
  change pid / tiles columns < (length+31)/32 at hd
  omega

@[simp] theorem float_read_written {α : Type} [Inhabited α] (dtype : FloatDType) (value : α) :
    toFloat dtype ((Cell.mk dtype.toTileDType (ofFloat dtype value)).read dtype.toTileDType) = value := by
  cases dtype <;> simp [toFloat,ofFloat]

theorem execution {α : Type} [Inhabited α] (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segment : Segment) (s : State α)
    (hstart : (s.mem p.starts p.batch).read .int = (segment.start : Int))
    (hlength : (s.mem p.lengths p.batch).read .int = (segment.length : Int))
    (hadapter : (s.mem p.adapters p.batch).read .int = (segment.adapter.getD 0 : Int))
    (fits : SegmentFits shape segment) (live : segment.active)
    (columns : p.columns = shape.columns) (slice : p.slice = shape.slice)
    (positive : 0 < shape.columns) (capacity : shape.slice+shape.columns ≤ shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride)
    (hp : s.pids 0 < tiles segment.length*tiles shape.columns) :
    ∃ final, exec (SourceBody.install M dot) (kernel abi p) s = some final ∧
      (∀ i, SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i →
        toFloat (SourceSuffix.storageType abi.output)
          ((final.mem p.output (SourceSuffix.offset p
            (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i)).read
            (SourceSuffix.storageType abi.output).toTileDType) =
          SourceSuffix.outputValues M abi p
            (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) s
            (SourceExecution.accumulated M dot abi p s segment.start (segment.adapter.getD 0) segment.length) i) ∧
      (∀ region address, (region ≠ p.output ∨ ∀ i,
        SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i →
          address ≠ SourceSuffix.offset p (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i) →
        final.mem region address = s.mem region address) := by
  obtain ⟨final,he,hm⟩ := SourceExecution.kernel_execution M dot abi p s segment.start
    (segment.adapter.getD 0) segment.length hstart hlength hadapter live.1
      (pid_guard p.columns segment.length (s.pids 0) (by omega) (by simpa only [columns] using hp))
  refine ⟨final,he,?_,?_⟩
  · intro i hi
    rw [hm,SourceSuffix.written_read _ _ _ _ _ _ i hi]
    · exact float_read_written _ _
    · exact SourceSuffix.offset_injective p _ shape.outputRows shape.outputColumns layout
        (fits.output live) (by simpa only [slice,columns] using capacity)
  · intro region address miss
    rw [hm]
    exact SourceSuffix.written_frame M abi p _ s _ region address miss

#axiomsClean pid_guard
#axiomsClean execution
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceTile
