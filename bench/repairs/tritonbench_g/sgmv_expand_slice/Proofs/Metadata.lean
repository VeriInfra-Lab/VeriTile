import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Geometry
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice

/-- Signed logical metadata returned by the three snapshot tolist calls. The
representation does not impose an unsigned interpretation on either width. -/
structure RawSegment where
  (start length adapter : Int)
  deriving DecidableEq, Repr

def RawSegment.decode (s : RawSegment) : Segment :=
  ⟨s.start.toNat,s.length.toNat,if s.adapter = -1 then none else some s.adapter.toNat⟩

/-- The exact two source validation guards, after the independently checked
tensor shapes and metadata lengths. N=0 does not bypass validation. -/
def RawSegment.accepted (p : Shape) (s : RawSegment) : Prop :=
  0 ≤ s.length ∧
    (s.length ≠ 0 ∧ s.adapter ≠ -1 →
      0 ≤ s.start ∧ s.start+s.length ≤ (min p.tokens p.outputRows : Nat) ∧
        0 ≤ s.adapter ∧ s.adapter < p.adapters)

instance (p : Shape) (s : RawSegment) : Decidable (s.accepted p) :=
  inferInstanceAs (Decidable (_ ∧ _))

def validateSegment (p : Shape) (s : RawSegment) : Option Segment :=
  if s.accepted p then some s.decode else none

def validateSegments (p : Shape) : List RawSegment → Option (List Segment)
  | [] => some []
  | s::ss => do
      let one ← validateSegment p s
      let rest ← validateSegments p ss
      pure (one::rest)

theorem decode_active (s : RawSegment) :
    s.decode.active ↔ 0 < s.length ∧ s.adapter ≠ -1 := by
  simp [RawSegment.decode,Segment.active]

theorem accepted_fits (p : Shape) (s : RawSegment) (h : s.accepted p) :
    SegmentFits p s.decode := by
  have hl := h.1
  constructor
  · intro ha
    have ha' := (decode_active s).mp ha
    obtain ⟨hstart,hend,_,_⟩ := h.2 ⟨by omega,ha'.2⟩
    change s.start.toNat+s.length.toNat ≤ p.tokens
    omega
  · intro ha
    have ha' := (decode_active s).mp ha
    obtain ⟨hstart,hend,_,_⟩ := h.2 ⟨by omega,ha'.2⟩
    change s.start.toNat+s.length.toNat ≤ p.outputRows
    omega
  · intro a hadapter hlen
    have hn : s.adapter ≠ -1 := by
      intro hn
      simp [RawSegment.decode,hn] at hadapter
    have hlen' : 0 < s.length := by
      change 0 < s.length.toNat at hlen
      omega
    obtain ⟨_,_,hnonneg,hbound⟩ := h.2 ⟨by omega,hn⟩
    have he : s.adapter.toNat = a := by
      simpa [RawSegment.decode,hn] using hadapter
    omega

theorem validateSegment_spec (p : Shape) (s : RawSegment) (out : Segment) :
    validateSegment p s = some out ↔ s.accepted p ∧ out = s.decode := by
  simp only [validateSegment]
  split_ifs <;> simp_all [eq_comm]

theorem validateSegments_spec (p : Shape) (raw : List RawSegment) (out : List Segment) :
    validateSegments p raw = some out ↔
      (∀ s ∈ raw, s.accepted p) ∧ out = raw.map RawSegment.decode := by
  induction raw generalizing out with
  | nil => simp [validateSegments,eq_comm]
  | cons s ss ih =>
      by_cases hs : s.accepted p
      · simp only [validateSegments,validateSegment,if_pos hs]
        cases hr : validateSegments p ss with
        | none =>
            have hbad : ¬∀ s ∈ ss, s.accepted p := by
              intro hall
              have good := (ih (ss.map RawSegment.decode)).mpr ⟨hall,rfl⟩
              rw [hr] at good
              contradiction
            simp [hbad,hs]
        | some tail =>
            obtain ⟨hall,rfl⟩ := (ih tail).mp hr
            have hall' : ∀ x ∈ s::ss, x.accepted p := by
              intro x hx
              rcases List.mem_cons.mp hx with rfl | hx
              · exact hs
              · exact hall x hx
            change (some (s.decode :: ss.map RawSegment.decode) = some out) ↔ _
            rw [Option.some.injEq]
            exact ⟨fun he => ⟨hall',he.symm⟩,fun he => he.2.symm⟩
      · simp [validateSegments,validateSegment,hs]

/-- Successful validation produces only ordinary fitted segments. Caller alias
and metadata storage properties belong to the preceding snapshot/tolist calls. -/
theorem validated_segments_fit (p : Shape) (raw : List RawSegment) (out : List Segment)
    (success : validateSegments p raw = some out) :
    out.length = raw.length ∧ ∀ s ∈ out, SegmentFits p s := by
  obtain ⟨all,rfl⟩ := (validateSegments_spec p raw out).mp success
  refine ⟨by simp,?_⟩
  intro s hs
  obtain ⟨source,hsource,rfl⟩ := List.mem_map.mp hs
  exact accepted_fits p source (all source hsource)

theorem negative_length_rejected (p : Shape) (s : RawSegment) (h : s.length < 0) :
    validateSegment p s = none := by
  simp [validateSegment,RawSegment.accepted,show ¬0 ≤ s.length by omega]

theorem sentinel_accepted (p : Shape) (s : RawSegment) (h : 0 ≤ s.length)
    (ha : s.adapter = -1) : validateSegment p s = some s.decode := by
  simp [validateSegment,RawSegment.accepted,h,ha]

theorem empty_accepted (p : Shape) (s : RawSegment) (h : s.length = 0) :
    validateSegment p s = some s.decode := by
  simp [validateSegment,RawSegment.accepted,h]

/-- Native sign extension preserves every metadata bit pattern, including the
sentinel and negative values that are subsequently rejected by validation. -/
theorem metadata32_widen (bits : BitVec 32) :
    (bits.signExtend 64).toInt = bits.toInt :=
  BitVec.toInt_signExtend_of_le (by decide)

#axiomsClean accepted_fits
#axiomsClean validateSegments_spec
#axiomsClean validated_segments_fit
#axiomsClean negative_length_rejected
#axiomsClean sentinel_accepted
#axiomsClean empty_accepted
#axiomsClean metadata32_widen
end VeriTile.Bench.Repairs.SgmvExpandSlice
