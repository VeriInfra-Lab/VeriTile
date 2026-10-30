import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.IO
import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Proofs.IndexKernel
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.PublicIO
import VeriTile.Triton.Float.ReadProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexPair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

/-- Exact local pair results compose through the original ordered jobs, retaining
overlapping segments and their metadata/read-region invariants. -/
theorem schedule_pair (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (a : SourceABI) (hi : a.input=.single) (hw : a.weight=.half) (ho : a.output=.single)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (separation : SourceGrid.PrivateRegions p) (fits : ∀ segment∈segments, SegmentFits shape segment)
    (inner : p.inner=shape.inner) (columns : p.columns=shape.columns) (slice : p.slice=shape.slice)
    (capacity : shape.slice+shape.columns≤shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride)
    (bounds : RegionName → Nat) (hb : PublicIO.DataBounds bounds p shape)
    (js : List Job) (s : State α) (infos : ∀ j∈js, PublicIO.JobInfo bounds p shape segments s j) :
    ScheduleSafe bounds (js.map (invocation (SourceBody.install M dot) a p)) s ∧
    runSchedule (js.map (invocation (SourceBody.install M dot) a p)) s =
      runSchedule (js.map (PublicIO.invocation (SourceBody.install M dot) a p)) s ∧
    ∃ final, runSchedule (js.map (PublicIO.invocation (SourceBody.install M dot) a p)) s=some final := by
  induction js generalizing s with
  | nil => exact ⟨trivial,rfl,s,rfl⟩
  | cons j js ih =>
    obtain ⟨segment,member,live,positive,pid,metadata,bs,bl,ba⟩ := infos j (by simp)
    let original := PublicIO.invocation (SourceBody.install M dot) a p j
    let candidate := invocation (SourceBody.install M dot) a p j
    let initial := original.initial s
    have hp : initial.pids 0=j.pid := by simp [initial,original,PublicIO.invocation,Invocation.initial,bindRegisters]
    have hstart : (initial.mem p.starts j.batch).read .int=(segment.start : Int) := metadata.start
    have hlength : (initial.mem p.lengths j.batch).read .int=(segment.length : Int) := metadata.length
    have hadapter : (initial.mem p.adapters j.batch).read .int=(segment.adapter.getD 0 : Int) := metadata.adapter
    have tile : initial.pids 0 / tiles p.columns * 32 < segment.length :=
      SourceTile.pid_guard p.columns segment.length (initial.pids 0)
        (by simpa only [columns] using positive) (by simpa only [hp,columns] using pid)
    have batchBounds : PublicIO.DataBounds bounds {p with batch := j.batch} shape :=
      ⟨hb.input,hb.weight,hb.output⟩
    have oneEq : candidate.run s=original.run s :=
      execution_eq M dot a hi hw ho {p with batch := j.batch} initial
        segment.start (segment.adapter.getD 0) segment.length hstart hlength hadapter live.1 tile
    have oneSafe : candidate.Safe bounds s := by
      apply kernel_safe M dot a {p with batch := j.batch} initial
        segment.start (segment.adapter.getD 0) segment.length hstart hlength hadapter live.1 tile bounds bs bl ba
      · intro block blockBound
        exact PublicIO.body_bounds bounds {p with batch := j.batch} shape batchBounds inner columns segment
          (fits segment member) live positive _ block blockBound
      · exact PublicIO.output_bounds bounds {p with batch := j.batch} shape batchBounds columns slice capacity segment
          (fits segment member) live _
    obtain ⟨next,oneRun,read,oneFrame⟩ := SourceTile.execution M dot a {p with batch := j.batch}
      shape segment initial hstart hlength hadapter (fits segment member) live columns slice positive
      capacity layout (by simpa only [hp] using pid)
    have originalRun : original.run s=some next := oneRun
    have candidateRun : candidate.run s=some next := oneEq.trans originalRun
    have readFrame : SourceGrid.ReadFrame p s next := by
      intro r hr offset
      exact oneFrame r offset (Or.inl hr)
    have tailInfos : ∀ q∈js, PublicIO.JobInfo bounds p shape segments next q := by
      intro q hq
      exact PublicIO.job_info_frame bounds p shape segments s next q separation readFrame
        (infos q (by simp [hq]))
    obtain ⟨tailSafe,tailEq,final,tailRun⟩ := ih next tailInfos
    refine ⟨?_,?_,final,?_⟩
    · change candidate.Safe bounds s ∧ ∀ t, candidate.run s=some t → _
      refine ⟨oneSafe,?_⟩
      intro t ht
      rw [candidateRun] at ht
      cases ht
      exact tailSafe
    · change (candidate.run s).bind _ = (original.run s).bind _
      rw [candidateRun,originalRun,Option.bind_some,Option.bind_some]
      exact tailEq
    · change (original.run s).bind _ = _
      rw [originalRun,Option.bind_some]
      exact tailRun

/-- Full original read-view IO. Execution, both source-safety traces and frames
are proved from metadata/input bounds, never supplied as caller assumptions. -/
theorem equivalent (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (a : SourceABI) (hi : a.input=.single) (hw : a.weight=.half) (ho : a.output=.single)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (separation : SourceGrid.PrivateRegions p) (fits : ∀ segment∈segments, SegmentFits shape segment)
    (inner : p.inner=shape.inner) (columns : p.columns=shape.columns) (slice : p.slice=shape.slice)
    (capacity : shape.slice+shape.columns≤shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride)
    (domain : ChannelValues α (PublicIO.inputChannels p shape segments) → Prop)
    (metadata : ∀ xs, domain xs → PublicIO.metadataDomain p shape segments xs) :
    (PublicIO.io (SourceBody.install M dot) a p shape segments).EquivalentOn
      (variant (SourceBody.install M dot) a p shape segments) domain := by
  intro xs hd bounds hb initial binding
  have metadataOk := PublicIO.binding_metadata (SourceBody.install M dot) a p shape segments xs
    (metadata xs hd) initial binding
  have infos := PublicIO.jobs_info bounds p shape segments 0 initial metadataOk
    (PublicIO.bounds_metadata (SourceBody.install M dot) a p shape segments xs bounds hb)
  have dataBounds := PublicIO.data_bounds (SourceBody.install M dot) a p shape segments xs bounds hb
  obtain ⟨safe,frame⟩ := PublicIO.schedule_safe_frame M dot a hw ho p shape segments separation fits
    inner columns slice capacity layout bounds dataBounds (jobs shape.columns segments) initial infos
  obtain ⟨candidateSafe,sameRun,final,ran⟩ := schedule_pair M dot a hi hw ho p shape segments
    separation fits inner columns slice capacity layout bounds dataBounds (jobs shape.columns segments) initial infos
  have exterior : (PublicIO.io (SourceBody.install M dot) a p shape segments).Frame xs initial final := by
    intro r offset outside
    exact frame final ran r offset
      (PublicIO.outside_footprint (SourceBody.install M dot) a p shape segments fits capacity xs r offset outside)
  refine ⟨safe,candidateSafe,final,final,ran,sameRun.trans ran,?_,exterior,exterior⟩
  intro o j active
  rfl

#axiomsClean schedule_pair
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexPair
