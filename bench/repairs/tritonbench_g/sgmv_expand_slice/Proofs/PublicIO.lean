import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.IOSchedule
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Shared IO composition: actual source access safety, successful full jobs,
all logical output reads, and the exact exterior cell frame. -/
theorem implements [Inhabited α] (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (abi : SourceABI) (hw : abi.weight=.half) (ho : abi.output=.single)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (separation : SourceGrid.PrivateRegions p)
    (fits : ∀ segment∈segments, SegmentFits shape segment)
    (inner : p.inner=shape.inner) (columns : p.columns=shape.columns) (slice : p.slice=shape.slice)
    (capacity : shape.slice+shape.columns≤shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride)
    (domain : ChannelValues α (inputChannels p shape segments) → Prop)
    (metadata : ∀ xs, domain xs → metadataDomain p shape segments xs)
    (formula : ChannelValues α (inputChannels p shape segments) → Nat → Nat → α)
    (result : ∀ xs, domain xs → ∀ initial,
      (io (SourceBody.install M dot) abi p shape segments).InputBinding xs initial →
      ∃ final, runSourceJobs (SourceBody.install M dot) abi p (jobs shape.columns segments) initial=some final ∧
        SourceSuffix.OutputMemory abi p shape final (formula xs)) :
    (io (SourceBody.install M dot) abi p shape segments).ImplementsOn domain
      (fun xs=>resultValues p shape (formula xs)) := by
  intro xs hd bounds hb initial binding
  have metadataOk := binding_metadata (SourceBody.install M dot) abi p shape segments xs
    (metadata xs hd) initial binding
  have infos := jobs_info bounds p shape segments 0 initial metadataOk
    (bounds_metadata (SourceBody.install M dot) abi p shape segments xs bounds hb)
  obtain ⟨safe,frame⟩ := schedule_safe_frame M dot abi hw ho p shape segments separation fits
    inner columns slice capacity layout bounds
    (data_bounds (SourceBody.install M dot) abi p shape segments xs bounds hb)
    (jobs shape.columns segments) initial infos
  refine ⟨safe,?_⟩
  obtain ⟨final,executed,output⟩ := result xs hd initial binding
  have run : runSchedule ((jobs shape.columns segments).map
      (invocation (SourceBody.install M dot) abi p)) initial=some final := by
    rw [schedule_run]
    exact executed
  refine ⟨final,run,output_readback (SourceBody.install M dot) abi ho p shape segments xs (formula xs) final output,?_⟩
  intro r a outside
  exact frame final run r a (outside_footprint (SourceBody.install M dot) abi p shape segments fits capacity xs r a outside)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
