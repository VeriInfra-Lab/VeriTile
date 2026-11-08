import bench.tritonbench_g.token_attn_llama2.IO

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.IOBinding
open VeriTile Triton FP.Structural Whole
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
variable {α β : Type} [Inhabited β]

/-- Only the finite declared input coordinates are bound; exterior cells and
all old output contents remain arbitrary. -/
structure Reads (c : Call) (m : Metadata c) (xs : Interface.Inputs α c)
    (encode : RegionName → Nat → α → β) (s : State β) : Prop where
  query : ∀ b h d, (s.mem "Q" (Reference.queryAddress c b.val h.val d.val)).read .real =
    encode "Q" (Reference.queryAddress c b.val h.val d.val) (Interface.query xs b h d)
  key : ∀ row h d, (s.mem "K" (Reference.keyAddress c row.val h.val d.val)).read .real =
    encode "K" (Reference.keyAddress c row.val h.val d.val) (Interface.key xs row h d)
  lengths : ∀ b : Fin c.batches, (s.mem "Length" b.val).read .int = m.lengths b
  starts : ∀ b : Fin c.batches, (s.mem "Start" b.val).read .int = m.starts b
  locations : ∀ o : Fin c.locSize, (s.mem "Loc" o.val).read .int = m.locations o

theorem reads (M : Program → Algebra β) (c : Call) (m : Metadata c) (scale : ℝ)
    (work : List Program) (encode : RegionName → Nat → α → β) (observe : β → α)
    (xs : Interface.Inputs α c) (s : State β) (hd : Interface.MetadataValues c m xs)
    (bound : (Interface.io M c m scale work encode observe).InputBinding xs s) :
    Reads c m xs encode s := by
  constructor
  · intro b h d
    exact bound ⟨0,by simp [Interface.io,Interface.inputs]⟩ (b,h,d,PUnit.unit) trivial
  · intro row h d
    exact bound ⟨1,by simp [Interface.io,Interface.inputs]⟩ (row,h,d,PUnit.unit) trivial
  · intro b
    exact (bound ⟨2,by simp [Interface.io,Interface.inputs]⟩ (b,PUnit.unit) trivial).trans (hd.1 b)
  · intro b
    exact (bound ⟨3,by simp [Interface.io,Interface.inputs]⟩ (b,PUnit.unit) trivial).trans (hd.2.1 b)
  · intro o
    exact (bound ⟨4,by simp [Interface.io,Interface.inputs]⟩ (o,PUnit.unit) trivial).trans (hd.2.2 o)

theorem Reads.preserve {c : Call} {m : Metadata c} {xs : Interface.Inputs α c}
    {encode : RegionName → Nat → α → β} {s t : State β} (scale : ℝ)
    (h : Reads c m xs encode s) (hr : ReadsEq (c.parameters scale) t s) :
    Reads c m xs encode t := by
  have q := hr "Q" (by simp [Call.parameters])
  have k := hr "K" (by simp [Call.parameters])
  have l := hr "Length" (by simp [Call.parameters])
  have a := hr "Start" (by simp [Call.parameters])
  have o := hr "Loc" (by simp [Call.parameters])
  constructor <;> intros <;> simp only [q,k,l,a,o] <;>
    first | apply h.query | apply h.key | apply h.lengths | apply h.starts | apply h.locations

theorem length_eq {c : Call} {m : Metadata c} {xs : Interface.Inputs α c}
    {encode : RegionName → Nat → α → β} {s : State β}
    (h : Reads c m xs encode s) (scale : ℝ) (pid : Program) (hb : pid.1<c.batches) :
    length (c.parameters scale) (start c.grid pid s) = m.len pid.1 := by
  simpa [length,lengthWord,Call.parameters,start,ids,Metadata.len,finiteRead,hb] using congrArg Int.toNat (h.lengths ⟨pid.1,hb⟩)

theorem length_word_eq {c : Call} {m : Metadata c} {xs : Interface.Inputs α c}
    {encode : RegionName → Nat → α → β} {s : State β}
    (h : Reads c m xs encode s) (scale : ℝ) (pid : Program) (hb : pid.1<c.batches) :
    lengthWord (c.parameters scale) (start c.grid pid s)=m.lengths ⟨pid.1,hb⟩ :=
  h.lengths ⟨pid.1,hb⟩

theorem window_eq {c : Call} {m : Metadata c} {xs : Interface.Inputs α c}
    {encode : RegionName → Nat → α → β} {s : State β}
    (h : Reads c m xs encode s) (v : Valid c m) (scale : ℝ)
    (pid : Program) (hb : pid.1<c.batches) (i : Index (c.parameters scale)) :
    windowIndex (c.parameters scale) (start c.grid pid s) i=
      (c.maxLength-m.len pid.1+(pid.2.2*32+i.1.val) : Nat) := by
  unfold windowIndex
  rw [length_word_eq h scale pid hb]
  simp only [Call.parameters]
  rw [←(signed_length c m v ⟨pid.1,hb⟩).2]
  simp [token,Call.parameters,start,ids,Int.natCast_add]

theorem active_eq {c : Call} {m : Metadata c} {xs : Interface.Inputs α c}
    {encode : RegionName → Nat → α → β} {s : State β}
    (h : Reads c m xs encode s) (v : Valid c m) (scale : ℝ)
    (pid : Program) (hb : pid.1<c.batches) (i : Index (c.parameters scale)) :
    active (c.parameters scale) (start c.grid pid s) i ↔
      pid.2.2*32+i.1.val < m.len pid.1 := by
  unfold active
  rw [window_eq h v scale pid hb i]
  have hl := length_bound c m v pid.1 hb
  change ((c.maxLength-m.len pid.1+(pid.2.2*32+i.1.val) : Nat) : Int)<(c.maxLength : Int) ↔ _
  omega

theorem offset_eq {c : Call} {m : Metadata c} {xs : Interface.Inputs α c}
    {encode : RegionName → Nat → α → β} {s : State β}
    (h : Reads c m xs encode s) (v : Valid c m) (scale : ℝ) (pid : Program)
    (hb : pid.1<c.batches) (i : Index (c.parameters scale)) :
    outputOffset (c.parameters scale) (start c.grid pid s) i =
      pid.2.1*c.outHead+(m.begin pid.1+(pid.2.2*32+i.1.val))*c.outSeq := by
  change ((pid.2.1*c.outHead : Nat)+((s.mem "Start" pid.1).read .int+_)*c.outSeq : Int).toNat=_
  rw [h.starts ⟨pid.1,hb⟩]
  have word : (m.begin pid.1 : Int)=m.starts ⟨pid.1,hb⟩ := by
    simp [Metadata.begin,finiteRead,hb,Int.toNat_of_nonneg (v.starts ⟨pid.1,hb⟩).1]
  rw [←word]
  simp only [token,Call.parameters,start,ids,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,↓reduceIte,reduceCtorEq,show (2 : Nat) ≠ 1 by decide]

theorem page_eq {c : Call} {m : Metadata c} {xs : Interface.Inputs α c}
    {encode : RegionName → Nat → α → β} {s : State β}
    (h : Reads c m xs encode s) (v : Valid c m) (scale : ℝ) (pid : Program)
    (hb : pid.1<c.batches) (i : Index (c.parameters scale))
    (hi : pid.2.2*32+i.1.val < m.len pid.1) :
    keyLocation (c.parameters scale) (start c.grid pid s) i =
      (m.page pid.1 (pid.2.2*32+i.1.val) : Int) := by
  have ha := (active_eq h v scale pid hb i).mpr hi
  have hl := length_bound c m v pid.1 hb
  have locBound := v.locationBounds pid.1 hb (c.maxLength-m.len pid.1+(pid.2.2*32+i.1.val))
    (by omega)
  simp only [keyLocation,ha,ite_true,window_eq h v scale pid hb i]
  have address : ((c.locBatch*pid.1 : Nat)+(c.locSeq : Int)*(c.maxLength-m.len pid.1+(pid.2.2*32+i.1.val) : Nat)).toNat=
      c.locBatch*pid.1+c.locSeq*(c.maxLength-m.len pid.1+(pid.2.2*32+i.1.val)) := by
    simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  change (s.mem "Loc" ((c.locBatch*pid.1 : Nat)+(c.locSeq : Int)*(c.maxLength-m.len pid.1+(pid.2.2*32+i.1.val) : Nat)).toNat).read .int=_
  rw [address,h.locations ⟨_,locBound⟩]
  simp [Metadata.page,finiteRead,locBound,Int.toNat_of_nonneg (v.pages ⟨pid.1,hb⟩ _ hi).1]
  simpa [finiteRead,locBound] using (v.pages ⟨pid.1,hb⟩ _ hi).1

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.IOBinding
