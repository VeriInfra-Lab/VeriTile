import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostFrame
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice

/-- Exactly the row/column destinations of one actual host slice copy. -/
def segmentCopyWrites (p : Shape) (v : OutputView) (s : Segment)
    (value : Nat → Nat → StoredWord v.width) : List ByteWrite :=
  if s.active then
    (List.range s.length).flatMap (fun r =>
      (List.range p.columns).flatMap (fun c =>
        cellWrites v (s.start+r) (p.slice+c) (value (s.start+r) (p.slice+c))))
  else []

/-- Batch-order copyback from the same final private output. Overlapping
segments may repeat a coordinate, always with that coordinate's final value. -/
def copyBackWrites (p : Shape) (v : OutputView) (segments : List Segment)
    (value : Nat → Nat → StoredWord v.width) : List ByteWrite :=
  segments.flatMap (fun s => segmentCopyWrites p v s value)

theorem segmentCopyWrites_authorized (p : Shape) (v : OutputView) (segments : List Segment)
    (s : Segment) (hs : s ∈ segments) (value : Nat → Nat → StoredWord v.width)
    (w : ByteWrite) (hw : w ∈ segmentCopyWrites p v s value) :
    AllowedByte p segments v w.backing w.offset := by
  unfold segmentCopyWrites at hw
  split_ifs at hw with active
  · obtain ⟨r,hr,hw⟩ := List.mem_flatMap.mp hw
    obtain ⟨c,hc,hw⟩ := List.mem_flatMap.mp hw
    refine cellWrites_authorized p segments v s hs (s.start+r) (p.slice+c) ?_
      (value (s.start+r) (p.slice+c)) w hw
    have hr' := List.mem_range.mp hr
    have hc' := List.mem_range.mp hc
    exact ⟨active,by omega,by omega,by omega,by omega⟩
  · simp at hw

theorem copyBackWrites_authorized (p : Shape) (v : OutputView) (segments : List Segment)
    (value : Nat → Nat → StoredWord v.width) (w : ByteWrite)
    (hw : w ∈ copyBackWrites p v segments value) : AllowedByte p segments v w.backing w.offset := by
  obtain ⟨s,hs,hw⟩ := List.mem_flatMap.mp hw
  exact segmentCopyWrites_authorized p v segments s hs value w hw

/-- Concrete successful copyback has the full caller-byte frame with no
whole-copy footprint premise. Values may be arbitrary, including all NaN bits. -/
theorem actual_copyback_frame (p : Shape) (v : OutputView) (segments : List Segment)
    (value : Nat → Nat → StoredWord v.width) (entry : Bytes) (a b : Nat)
    (outside : ¬ AllowedByte p segments v a b) :
    (Host.returned v (copyBackWrites p v segments value) entry).memory a b = entry a b :=
  Host.successful_frame p segments v _ entry
    (copyBackWrites_authorized p v segments value) a b outside

/-- Ordinary failure-local copy contract: any committed byte has a destination
of the actual requested copies. It need not have the final expected byte value
or occur in source order. -/
def PartialCopyFootprint (committed requested : List ByteWrite) : Prop :=
  ∀ w ∈ committed, ∃ destination ∈ requested,
    w.backing = destination.backing ∧ w.offset = destination.offset

theorem actual_failed_copy_frame (p : Shape) (v : OutputView) (segments : List Segment)
    (value : Nat → Nat → StoredWord v.width) (entry : Bytes) (stage : Host.Stage)
    (committed : List ByteWrite)
    (localCopy : PartialCopyFootprint committed (copyBackWrites p v segments value))
    (a b : Nat) (outside : ¬ AllowedByte p segments v a b) :
    (Host.failed v stage committed entry).memory a b = entry a b := by
  apply Host.failed_copy_frame p segments v stage committed entry _ a b outside
  intro w hw
  obtain ⟨destination,hd,ha,hb⟩ := localCopy w hw
  rw [ha,hb]
  exact copyBackWrites_authorized p v segments value destination hd

/-- Dense logical coordinates have distinct physical bytes. Singleton axes are
valid: their only coordinate is zero, so their ignored original strides do not
enter this dense representation of the contiguous output view. -/
theorem output_byte_injective (v : OutputView) {r c b r' c' b' : Nat}
    (hc : c < v.columns) (hc' : c' < v.columns)
    (hb : b < v.width) (hb' : b' < v.width)
    (he : v.byte r c b = v.byte r' c' b') : r = r' ∧ c = c' ∧ b = b' := by
  obtain ⟨he,heByte⟩ := linear_injective hb hb' he
  have heCell : r*v.columns+c = r'*v.columns+c' := by
    unfold OutputView.byte at *
    omega
  obtain ⟨heRow,heColumn⟩ := linear_injective hc hc' heCell
  exact ⟨heRow,heColumn,heByte⟩

theorem copyBackWrites_member (p : Shape) (v : OutputView) (segments : List Segment)
    (value : Nat → Nat → StoredWord v.width) (w : ByteWrite)
    (hw : w ∈ copyBackWrites p v segments value) :
    ∃ s ∈ segments, ∃ r c, ∃ b : Fin v.width, ActiveCell p s r c ∧
      w = ⟨v.backing,v.byte r c b.val,value r c b⟩ := by
  obtain ⟨s,hs,hw⟩ := List.mem_flatMap.mp hw
  unfold segmentCopyWrites at hw
  split_ifs at hw with active
  · obtain ⟨r,hr,hw⟩ := List.mem_flatMap.mp hw
    obtain ⟨c,hc,hw⟩ := List.mem_flatMap.mp hw
    obtain ⟨b,_,rfl⟩ := List.mem_map.mp hw
    refine ⟨s,hs,s.start+r,p.slice+c,b,?_,rfl⟩
    have hr' := List.mem_range.mp hr
    have hc' := List.mem_range.mp hc
    exact ⟨active,by omega,by omega,by omega,by omega⟩
  · simp at hw

theorem active_byte_in_copyBack (p : Shape) (v : OutputView) (segments : List Segment)
    (value : Nat → Nat → StoredWord v.width) (s : Segment) (hs : s ∈ segments)
    (r c : Nat) (active : ActiveCell p s r c) (b : Fin v.width) :
    (⟨v.backing,v.byte r c b.val,value r c b⟩ : ByteWrite) ∈
      copyBackWrites p v segments value := by
  apply List.mem_flatMap.mpr
  refine ⟨s,hs,?_⟩
  unfold segmentCopyWrites
  rw [if_pos active.1]
  apply List.mem_flatMap.mpr
  refine ⟨r-s.start,List.mem_range.mpr (by have := active.2.1; have := active.2.2.1; omega),?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨c-p.slice,List.mem_range.mpr (by have := active.2.2.2.1; have := active.2.2.2.2; omega),?_⟩
  have hr : s.start+(r-s.start)=r := by have := active.2.1; omega
  have hc : p.slice+(c-p.slice)=c := by have := active.2.2.2.1; omega
  rw [hr,hc]
  exact List.mem_map.mpr ⟨b,by simp,rfl⟩

/-- Repeated writes of a byte agree, regardless of multiplicity and order. -/
theorem writeAll_uniform (writes : List ByteWrite) (entry : Bytes) (a b : Nat)
    (expected : BitVec 8)
    (uniform : ∀ w ∈ writes, a = w.backing → b = w.offset → w.value = expected)
    (present : (∃ w ∈ writes, a = w.backing ∧ b = w.offset) ∨ entry a b = expected) :
    writeAll writes entry a b = expected := by
  induction writes generalizing entry with
  | nil => simpa [writeAll] using present
  | cons w ws ih =>
      change writeAll ws (w.apply entry) a b = expected
      apply ih _ (by intro x hx; exact uniform x (List.mem_cons_of_mem _ hx))
      by_cases hit : a = w.backing ∧ b = w.offset
      · right
        simpa [ByteWrite.apply,hit] using uniform w (by simp) hit.1 hit.2
      · rcases present with ⟨x,hx,hxa,hxb⟩ | he
        · left
          refine ⟨x,?_,hxa,hxb⟩
          rcases List.mem_cons.mp hx with rfl | hx
          · exact (hit ⟨hxa,hxb⟩).elim
          · exact hx
        · right
          simpa [ByteWrite.apply,hit] using he

/-- Every active byte equals the final private-output value. Overlapping
segments may copy the same coordinate repeatedly; coordinate injectivity and
the shared final value ensure that every such write agrees. The value argument
is the caller storage representation, including any output-view negation. -/
theorem actual_copyback_readback (p : Shape) (v : OutputView) (segments : List Segment)
    (capacity : p.slice+p.columns ≤ v.columns)
    (value : Nat → Nat → StoredWord v.width) (entry : Bytes)
    (s : Segment) (hs : s ∈ segments) (r c : Nat) (active : ActiveCell p s r c)
    (b : Fin v.width) :
    (Host.returned v (copyBackWrites p v segments value) entry).memory
      v.backing (v.byte r c b.val) = value r c b := by
  apply writeAll_uniform
  · intro w hw _ he
    obtain ⟨s',_,r',c',b',ha,rfl⟩ := copyBackWrites_member p v segments value w hw
    have hc : c < v.columns := by have := active.2.2.2.2; omega
    have hc' : c' < v.columns := by have := ha.2.2.2.2; omega
    obtain ⟨rfl,rfl,hb⟩ := output_byte_injective v hc hc' b.isLt b'.isLt he
    have hb' : b = b' := Fin.ext hb
    subst b'
    rfl
  · left
    exact ⟨_,active_byte_in_copyBack p v segments value s hs r c active b,rfl,rfl⟩

#axiomsClean actual_copyback_frame
#axiomsClean actual_failed_copy_frame
#axiomsClean output_byte_injective
#axiomsClean actual_copyback_readback
end VeriTile.Bench.Repairs.SgmvExpandSlice
