import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.IOAccess

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

theorem metadataList_of_indexed [Inhabited α] (p : KernelArgs) (segments : List Segment)
    (first : Nat) (s : State α)
    (indexed : ∀ j : Fin segments.length, (segments[j]).active →
      SourceGrid.Metadata {p with batch := first+j.val} (segments[j]) s) :
    SourceGrid.MetadataList p first segments s := by
  induction segments generalizing first with
  | nil => trivial
  | cons segment tail ih =>
    constructor
    · simpa using indexed ⟨0,by simp⟩
    · apply ih (first+1)
      intro j live
      simpa [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using indexed ⟨j.val+1,by simpa using j.isLt⟩ live

theorem metadataBounds_of_indexed (bounds : RegionName → Nat) (p : KernelArgs)
    (segments : List Segment) (first : Nat)
    (indexed : ∀ j : Fin segments.length, (segments[j]).active →
      first+j.val<bounds p.starts ∧ first+j.val<bounds p.lengths ∧ first+j.val<bounds p.adapters) :
    MetadataBounds bounds p first segments := by
  induction segments generalizing first with
  | nil => trivial
  | cons segment tail ih =>
    constructor
    · simpa using indexed ⟨0,by simp⟩
    · apply ih (first+1)
      intro j live
      simpa [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using indexed ⟨j.val+1,by simpa using j.isLt⟩ live

theorem binding_metadata [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (xs : ChannelValues α (inputChannels p shape segments)) (domain : metadataDomain p shape segments xs)
    (s : State α) (binding : (io M abi p shape segments).InputBinding xs s) :
    SourceGrid.MetadataList p 0 segments s := by
  apply metadataList_of_indexed
  intro j live
  obtain ⟨hs,hl,ha⟩ := domain j live
  simp only [Nat.zero_add]
  constructor
  · exact (binding ⟨3,by simp [io]⟩ (j,PUnit.unit) live).trans hs
  · exact (binding ⟨4,by simp [io]⟩ (j,PUnit.unit) live).trans hl
  · exact (binding ⟨5,by simp [io]⟩ (j,PUnit.unit) live).trans ha

theorem bounds_metadata (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (xs : ChannelValues α (inputChannels p shape segments))
    (bounds : RegionName → Nat) (hb : (io M abi p shape segments).InBounds bounds xs) :
    MetadataBounds bounds p 0 segments := by
  apply metadataBounds_of_indexed
  intro j live
  simp only [Nat.zero_add]
  exact ⟨hb.1.2.1 ⟨3,by simp [io]⟩ (j,PUnit.unit) live,
    hb.1.2.1 ⟨4,by simp [io]⟩ (j,PUnit.unit) live,
    hb.1.2.1 ⟨5,by simp [io]⟩ (j,PUnit.unit) live⟩

theorem outside_footprint (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (fits : ∀ segment∈segments, SegmentFits shape segment)
    (capacity : shape.slice+shape.columns≤shape.outputColumns)
    (xs : ChannelValues α (inputChannels p shape segments))
    (r : RegionName) (a : Nat)
    (outside : (io M abi p shape segments).toProgramIO.Outside xs r a) :
    ¬Footprint p shape segments r a := by
  rintro ⟨region,segment,member,row,col,active,offset⟩
  have hr : row<shape.outputRows := lt_of_lt_of_le active.2.2.1 ((fits segment member).output active.1)
  have hc : col<shape.outputColumns := lt_of_lt_of_le active.2.2.2.2 capacity
  have miss := outside.1 ⟨0,by simp [io]⟩ (⟨row,hr⟩,⟨col,hc⟩,PUnit.unit) ⟨segment,member,active⟩
  exact miss.elim (fun no=>no region) (fun no=>no offset)

theorem output_readback [Inhabited α] (M : Algebra α) (abi : SourceABI) (ho : abi.output=.single)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (xs : ChannelValues α (inputChannels p shape segments)) (f : Nat → Nat → α)
    (s : State α) (output : SourceSuffix.OutputMemory abi p shape s f) :
    (io M abi p shape segments).OutputReadback xs (resultValues p shape f) s := by
  intro o j active
  have ofact : o=⟨0,by simp [io]⟩ := Fin.eq_zero o
  subst o
  have got := output j.1.val j.2.1.val j.1.isLt j.2.1.isLt
  unfold SourceSuffix.readOutput at got
  rw [ho] at got
  exact got

#axiomsClean binding_metadata
#axiomsClean bounds_metadata
#axiomsClean outside_footprint
#axiomsClean output_readback
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
