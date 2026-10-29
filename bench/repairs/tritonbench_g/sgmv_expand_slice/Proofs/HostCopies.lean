import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.CopyBack

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.HostCopies

/-- An ordinary local copy may fail after writing any bytes in its requested
destination. Successful copy means exactly the requested local byte writes. -/
structure API where
  call : (batch : Nat) → (requested : List ByteWrite) →
    Except {committed : List ByteWrite // PartialCopyFootprint committed requested} Unit

structure Record where
  outcome : Host.Outcome
  writes : List ByteWrite

def run (api : API) (shape : Shape) (view : OutputView)
    (value : Nat → Nat → StoredWord view.width) : Nat → List Segment → Record
  | _,[] => ⟨.returnedNone,[]⟩
  | batch,segment::segments =>
      if segment.active ∧ 0 < shape.columns then
        let requested := segmentCopyWrites shape view segment value
        match api.call batch requested with
        | .error committed => ⟨.failed (.copy batch),committed.val⟩
        | .ok _ =>
            let tail := run api shape view value (batch+1) segments
            ⟨tail.outcome,requested++tail.writes⟩
      else run api shape view value (batch+1) segments

def result (view : OutputView) (bytes : Bytes) (record : Record) : Host.Result :=
  ⟨record.outcome,view.object,view.backing,writeAll record.writes bytes⟩

theorem empty_copy (shape : Shape) (view : OutputView) (segment : Segment)
    (value : Nat → Nat → StoredWord view.width) (inactive : ¬(segment.active ∧ 0 < shape.columns)) :
    segmentCopyWrites shape view segment value = [] := by
  by_cases active : segment.active
  · have hn : ¬0 < shape.columns := fun h => inactive ⟨active,h⟩
    have hz : shape.columns = 0 := by omega
    simp [segmentCopyWrites,active,hz]
  · simp [segmentCopyWrites,active]

theorem authorized (api : API) (shape : Shape) (view : OutputView)
    (value : Nat → Nat → StoredWord view.width) (batch : Nat) (segments : List Segment)
    (w : ByteWrite) (member : w ∈ (run api shape view value batch segments).writes) :
    AllowedByte shape segments view w.backing w.offset := by
  induction segments generalizing batch with
  | nil => simp [run] at member
  | cons segment segments ih =>
      unfold run at member
      by_cases active : segment.active ∧ 0 < shape.columns
      · rw [if_pos active] at member
        cases hc : api.call batch (segmentCopyWrites shape view segment value) with
        | error committed =>
            simp only [hc] at member
            obtain ⟨destination,hd,ha,hb⟩ := committed.property w member
            rw [ha,hb]
            exact segmentCopyWrites_authorized shape view (segment::segments) segment (by simp) value destination hd
        | ok unit =>
            simp only [hc,List.mem_append] at member
            rcases member with head | tail
            · exact segmentCopyWrites_authorized shape view (segment::segments) segment (by simp) value w head
            · obtain ⟨backing,s,hs,r,c,b,ha,hb,he⟩ := ih (batch+1) tail
              exact ⟨backing,s,List.mem_cons_of_mem _ hs,r,c,b,ha,hb,he⟩
      · rw [if_neg active] at member
        obtain ⟨backing,s,hs,r,c,b,ha,hb,he⟩ := ih (batch+1) member
        exact ⟨backing,s,List.mem_cons_of_mem _ hs,r,c,b,ha,hb,he⟩

theorem frame (api : API) (shape : Shape) (view : OutputView)
    (value : Nat → Nat → StoredWord view.width) (batch : Nat) (segments : List Segment)
    (bytes : Bytes) (a b : Nat) (outside : ¬AllowedByte shape segments view a b) :
    (result view bytes (run api shape view value batch segments)).memory a b = bytes a b :=
  authorized_copy_frame shape segments view _ bytes (authorized api shape view value batch segments) a b outside

theorem complete (api : API) (shape : Shape) (view : OutputView)
    (value : Nat → Nat → StoredWord view.width) (batch : Nat) (segments : List Segment)
    (available : ∀ batch requested, api.call batch requested = .ok ()) :
    run api shape view value batch segments = ⟨.returnedNone,copyBackWrites shape view segments value⟩ := by
  induction segments generalizing batch with
  | nil => rfl
  | cons segment segments ih =>
      unfold run
      by_cases active : segment.active ∧ 0 < shape.columns
      · rw [if_pos active]
        simp only [available,ih,copyBackWrites,List.flatMap_cons]
      · rw [if_neg active,ih]
        simp only [copyBackWrites,List.flatMap_cons,empty_copy shape view segment value active,List.nil_append]

#axiomsClean empty_copy
#axiomsClean authorized
#axiomsClean frame
#axiomsClean complete
end VeriTile.Bench.Repairs.SgmvExpandSlice.HostCopies
