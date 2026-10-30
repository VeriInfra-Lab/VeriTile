import bench.tritonbench_g.sgmv_expand_slice.Proofs.IOAccess

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem job_info_frame [Inhabited α] (bounds : RegionName → Nat) (p : KernelArgs) (shape : Shape)
    (segments : List Segment) (s t : State α) (j : Job)
    (separation : SourceGrid.PrivateRegions p) (frame : SourceGrid.ReadFrame p s t)
    (info : JobInfo bounds p shape segments s j) : JobInfo bounds p shape segments t j := by
  obtain ⟨segment,member,live,positive,pid,metadata,bs,bl,ba⟩ := info
  exact ⟨segment,member,live,positive,pid,
    SourceGrid.metadata_frame {p with batch := j.batch} segment s t metadata
      (SourceGrid.separation_batch p j.batch separation) frame,bs,bl,ba⟩

theorem local_footprint (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (segment : Segment) (member : segment∈segments) (live : segment.active)
    (columns : p.columns=shape.columns) (slice : p.slice=shape.slice)
    (pid : Nat) (i : TileIndex [32,32])
    (active : SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length pid) i) :
    Footprint p shape segments p.output
      (SourceSuffix.offset p (SourceExecution.outputCoordinates p segment.start segment.length pid) i) := by
  refine ⟨rfl,segment,member,
    SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length pid) i,
    SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length pid) i,?_,
    SourceSuffix.active_offset _ _ _ active⟩
  refine ⟨live,?_,active.1,?_,?_⟩
  · simp only [SourceSuffix.row,SourceExecution.outputCoordinates]; omega
  · simp only [SourceSuffix.column,SourceExecution.outputCoordinates,slice]; omega
  · simpa only [columns,slice] using active.2

/-- Every actual source invocation is safe, and its exact raw cell frame
composes through the existing ordered job schedule. -/
theorem schedule_safe_frame [Inhabited α] (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (abi : SourceABI) (hw : abi.weight=.half) (ho : abi.output=.single)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (separation : SourceGrid.PrivateRegions p)
    (fits : ∀ segment∈segments, SegmentFits shape segment)
    (inner : p.inner=shape.inner) (columns : p.columns=shape.columns) (slice : p.slice=shape.slice)
    (capacity : shape.slice+shape.columns≤shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride)
    (bounds : RegionName → Nat) (hb : DataBounds bounds p shape)
    (js : List Job) (s : State α)
    (infos : ∀ j∈js, JobInfo bounds p shape segments s j) :
    ScheduleSafe bounds (js.map (invocation (SourceBody.install M dot) abi p)) s ∧
    ∀ final, runSchedule (js.map (invocation (SourceBody.install M dot) abi p)) s=some final →
      ∀ r a, ¬Footprint p shape segments r a → final.mem r a=s.mem r a := by
  induction js generalizing s with
  | nil =>
    refine ⟨trivial,?_⟩
    intro final run
    cases run
    intros
    rfl
  | cons j js ih =>
    obtain ⟨segment,member,live,positive,pid,metadata,bs,bl,ba⟩ := infos j (by simp)
    let job := invocation (SourceBody.install M dot) abi p j
    let initial := job.initial s
    have hp : initial.pids 0=j.pid := by simp [initial,job,invocation,Invocation.initial,bindRegisters]
    have hstart : (initial.mem p.starts j.batch).read .int=(segment.start : Int) := metadata.start
    have hlength : (initial.mem p.lengths j.batch).read .int=(segment.length : Int) := metadata.length
    have hadapter : (initial.mem p.adapters j.batch).read .int=(segment.adapter.getD 0 : Int) := metadata.adapter
    have batchBounds : DataBounds bounds {p with batch := j.batch} shape :=
      ⟨hb.input,hb.weight,hb.output⟩
    have oneSafe : job.Safe bounds s := by
      apply SourceAccess.kernel_safe M dot abi hw ho {p with batch := j.batch} initial
        segment.start (segment.adapter.getD 0) segment.length hstart hlength hadapter live.1
      · exact SourceTile.pid_guard p.columns segment.length (initial.pids 0)
          (by simpa only [columns] using positive) (by simpa only [hp,columns] using pid)
      · exact bs
      · exact bl
      · exact ba
      · intro block blockBound
        exact body_bounds bounds {p with batch := j.batch} shape batchBounds inner columns segment
          (fits segment member) live positive _ block blockBound
      · exact output_bounds bounds {p with batch := j.batch} shape batchBounds columns slice capacity segment
          (fits segment member) live _
    obtain ⟨next,oneRun,read,oneFrame⟩ := SourceTile.execution M dot abi {p with batch := j.batch}
      shape segment initial hstart hlength hadapter (fits segment member) live columns slice positive
      capacity layout (by simpa only [hp] using pid)
    have jobRun : job.run s=some next := oneRun
    have readFrame : SourceGrid.ReadFrame p s next := by
      intro r hr a
      exact oneFrame r a (Or.inl hr)
    have tailInfo : ∀ q∈js, JobInfo bounds p shape segments next q := by
      intro q hq
      exact job_info_frame bounds p shape segments s next q separation readFrame
        (infos q (by simp [hq]))
    obtain ⟨tailSafe,tailFrame⟩ := ih next tailInfo
    constructor
    · change job.Safe bounds s ∧ ∀ t, job.run s=some t → _
      refine ⟨oneSafe,?_⟩
      intro t ht
      rw [jobRun] at ht
      cases ht
      exact tailSafe
    · intro final run r a outside
      change (job.run s).bind _=some final at run
      rw [jobRun,Option.bind_some] at run
      rw [tailFrame final run r a outside]
      apply oneFrame r a
      by_cases hr : r=p.output
      · right
        intro i active eq
        apply outside
        rw [hr,eq]
        exact local_footprint {p with batch := j.batch} shape segments segment member live columns slice
          (initial.pids 0) i active
      · exact Or.inl hr

#axiomsClean schedule_safe_frame
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
