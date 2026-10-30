import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.PreparedMemory

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PreparedMemory
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1400000
set_option maxRecDepth 20000

theorem metadataList_iff {α : Type} [Inhabited α] (p : KernelArgs) (first : Nat)
    (segments : List Segment) (s : State α) :
    SourceGrid.MetadataList p first segments s ↔
      ∀ j segment, segments[j]? = some segment → segment.active →
        SourceGrid.Metadata {p with batch := first+j} segment s := by
  induction segments generalizing first with
  | nil => simp [SourceGrid.MetadataList]
  | cons head tail ih =>
      constructor
      · intro h j segment hj live
        cases j with
        | zero =>
            have he : head = segment := by simpa using hj
            subst segment
            simpa using h.1 live
        | succ j =>
            have ht : tail[j]? = some segment := by simpa using hj
            simpa only [Nat.add_assoc,Nat.add_comm 1 j] using (ih (first+1)).mp h.2 j segment ht live
      · intro h
        refine ⟨?_,(ih (first+1)).mpr ?_⟩
        · intro live
          simpa using h 0 head rfl live
        · intro j segment hj live
          simpa only [Nat.add_assoc,Nat.add_comm 1 j] using h (j+1) segment hj live

theorem decoded_metadata {α : Type} [Inhabited α] (abi : SourceABI) (shape : Shape) (strides : Strides)
    (X : Nat → Nat → α) (W : Nat → Nat → Nat → α) (Y : Nat → Nat → α)
    (raw : List RawSegment) (extraLengths : List Int) (j : Nat) (source : RawSegment)
    (get : raw[j]? = some source) (accepted : source.accepted shape) (live : source.decode.active) :
    SourceGrid.Metadata {(arguments shape strides) with batch := j} source.decode
      (state abi shape X W Y raw extraLengths) := by
  have hi : j < raw.length := (List.getElem?_eq_some_iff.mp get).choose
  have hlength : ((raw.map RawSegment.length)++extraLengths)[j]? = some source.length := by
    rw [List.getElem?_append_left (by simpa using hi),List.getElem?_map,get]
    rfl
  have active := (decode_active source).mp live
  have valid := accepted.2 ⟨by omega,active.2⟩
  constructor
  all_goals simp [state,arguments,get,hlength,RawSegment.decode,active.2]
  case start => exact valid.1
  case length => exact accepted.1
  case adapter => exact valid.2.2.1

/-- Validation and the source loads read the same private metadata snapshots.
The unused suffix of the lengths snapshot is retained without restriction. -/
theorem validated_metadata {α : Type} [Inhabited α] (abi : SourceABI) (shape : Shape) (strides : Strides)
    (X : Nat → Nat → α) (W : Nat → Nat → Nat → α) (Y : Nat → Nat → α)
    (raw : List RawSegment) (extraLengths : List Int) (segments : List Segment)
    (validated : validateSegments shape raw = some segments) :
    SourceGrid.MetadataList (arguments shape strides) 0 segments (state abi shape X W Y raw extraLengths) := by
  obtain ⟨all,rfl⟩ := (validateSegments_spec shape raw segments).mp validated
  apply (metadataList_iff _ _ _ _).mpr
  intro j segment hget live
  rw [List.getElem?_map] at hget
  cases hr : raw[j]? with
  | none => simp [hr] at hget
  | some source =>
      have he : source.decode = segment := by simpa [hr] using hget
      subst segment
      simpa only [Nat.zero_add] using decoded_metadata abi shape strides X W Y raw extraLengths j source hr
        (all source (List.mem_of_getElem? hr)) live

#axiomsClean metadataList_iff
#axiomsClean decoded_metadata
#axiomsClean validated_metadata
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.PreparedMemory
