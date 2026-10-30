import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.IOLayout
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceAccessKernel

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

structure DataBounds (bounds : RegionName → Nat) (p : KernelArgs) (shape : Shape) : Prop where
  input : ∀ r k, r<shape.tokens → k<shape.inner →
    r*p.inputRowStride+k*p.inputInnerStride < bounds p.input
  weight : ∀ a c k, a<shape.adapters → c<shape.columns → k<shape.inner →
    p.adapterStride*a+k*p.weightInnerStride+c*p.weightColumnStride < bounds p.weight
  output : ∀ r c, r<shape.outputRows → c<shape.outputColumns →
    r*p.outputRowStride+c*p.outputColumnStride < bounds p.output

theorem data_bounds (M : Algebra α) (abi : SourceABI) (p : KernelArgs) (shape : Shape)
    (segments : List Segment) (xs : ChannelValues α (inputChannels p shape segments))
    (bounds : RegionName → Nat) (hb : (io M abi p shape segments).InBounds bounds xs) :
    DataBounds bounds p shape := by
  constructor
  · intro r k hr hk
    exact hb.1.2.1 ⟨0,by simp [io]⟩ (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) trivial
  · intro a c k ha hc hk
    exact hb.1.2.1 ⟨1,by simp [io]⟩ (⟨a,ha⟩,⟨c,hc⟩,⟨k,hk⟩,PUnit.unit) trivial
  · intro r c hr hc
    exact hb.1.2.1 ⟨2,by simp [io]⟩ (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) trivial

theorem body_bounds (bounds : RegionName → Nat) (p : KernelArgs) (shape : Shape)
    (hb : DataBounds bounds p shape) (inner : p.inner=shape.inner)
    (columns : p.columns=shape.columns) (segment : Segment) (fits : SegmentFits shape segment)
    (live : segment.active) (positive : 0<shape.columns) (pid block : Nat)
    (blockBound : block<(p.inner+15)/16) :
    SourceAccess.BodyBounds bounds p
      (SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length pid) block := by
  constructor
  · intro i active
    have hk := (SourceMemory.exact_mask p.inner block i.2.1 blockBound).mp active
    have hr := SourceMemory.coordinates_rows p shape segment fits live pid i.1
    have got := hb.input _ _ hr (by simpa only [inner] using hk)
    simpa [SourceBody.aOffset,SourceBody.safeInner,SourceBody.globalInner,hk,
      SourceInit.coordinates,Nat.add_mul] using got
  · intro i active
    have hk := (SourceMemory.exact_mask p.inner block i.1 blockBound).mp active
    have ha := SourceMemory.coordinates_adapter shape segment fits live
    have hc := SourceMemory.coordinates_columns p shape segment columns positive pid i.2.1
    have got := hb.weight _ _ _ ha hc (by simpa only [inner] using hk)
    simpa [SourceBody.bOffset,SourceBody.safeInner,SourceBody.globalInner,hk,SourceInit.coordinates] using got

theorem output_bounds (bounds : RegionName → Nat) (p : KernelArgs) (shape : Shape)
    (hb : DataBounds bounds p shape) (columns : p.columns=shape.columns) (slice : p.slice=shape.slice)
    (capacity : shape.slice+shape.columns≤shape.outputColumns)
    (segment : Segment) (fits : SegmentFits shape segment) (live : segment.active)
    (pid : Nat) (i : TileIndex [32,32])
    (active : SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length pid) i) :
    SourceSuffix.offset p (SourceExecution.outputCoordinates p segment.start segment.length pid) i < bounds p.output := by
  rw [SourceSuffix.active_offset _ _ _ active]
  apply hb.output
  · exact lt_of_lt_of_le active.1 (fits.output live)
  · exact lt_of_lt_of_le active.2 (by simpa only [columns,slice] using capacity)

def Footprint (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (region : RegionName) (offset : Nat) : Prop :=
  region=p.output ∧ ∃ segment ∈ segments, ∃ r c, ActiveCell shape segment r c ∧
    offset=r*p.outputRowStride+c*p.outputColumnStride

def MetadataBounds (bounds : RegionName → Nat) (p : KernelArgs) : Nat → List Segment → Prop
  | _,[] => True
  | first,segment::tail =>
      (segment.active → first<bounds p.starts ∧ first<bounds p.lengths ∧ first<bounds p.adapters) ∧
      MetadataBounds bounds p (first+1) tail

def JobInfo [Inhabited α] (bounds : RegionName → Nat) (p : KernelArgs) (shape : Shape)
    (segments : List Segment) (s : State α) (j : Job) : Prop :=
  ∃ segment ∈ segments, segment.active ∧ 0<shape.columns ∧
    j.pid<tiles segment.length*tiles shape.columns ∧
    SourceGrid.Metadata {p with batch := j.batch} segment s ∧
    j.batch<bounds p.starts ∧ j.batch<bounds p.lengths ∧ j.batch<bounds p.adapters

theorem jobs_info [Inhabited α] (bounds : RegionName → Nat) (p : KernelArgs) (shape : Shape)
    (segments : List Segment) (first : Nat) (s : State α)
    (metadata : SourceGrid.MetadataList p first segments s)
    (hb : MetadataBounds bounds p first segments) :
    ∀ j ∈ jobsFrom shape.columns first segments, JobInfo bounds p shape segments s j := by
  induction segments generalizing first with
  | nil => simp [jobsFrom]
  | cons segment tail ih =>
    intro j hj
    simp only [jobsFrom,List.mem_append] at hj
    rcases hj with hj | hj
    · unfold batchJobs at hj
      split_ifs at hj with active
      · obtain ⟨pid,hpid,rfl⟩ := List.mem_map.mp hj
        exact ⟨segment,by simp,active.1,active.2,List.mem_range.mp hpid,metadata.1 active.1,hb.1 active.1⟩
      · simp at hj
    · obtain ⟨seg,hs,live,positive,hpid,hm,bs,bl,ba⟩ := ih (first+1) metadata.2 hb.2 j hj
      exact ⟨seg,List.mem_cons_of_mem _ hs,live,positive,hpid,hm,bs,bl,ba⟩

#axiomsClean data_bounds
#axiomsClean body_bounds
#axiomsClean output_bounds
#axiomsClean jobs_info
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
