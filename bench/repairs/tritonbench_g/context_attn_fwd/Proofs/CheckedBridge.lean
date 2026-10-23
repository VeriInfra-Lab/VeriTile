import bench.repairs.tritonbench_g.context_attn_fwd.Contract
import bench.repairs.tritonbench_g.context_attn_fwd.Misc.Empty

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Checked
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- Ordinary CPU int32 observations of the original contiguous metadata
bytes. No host or kernel result appears in this interface. -/
structure MetadataObservation (c : Host.Call) (B : Native.Bytes extent) (m : Physical.Memory)
    (start sequence prompt : Nat → Int) : Prop where
  start : ∀ b,b<c.batches → (Snapshot.load32 m (B.base c.regions.start+4*b)).toInt=start b
  sequence : ∀ b,b<c.batches → (Snapshot.load32 m (B.base c.regions.sequence+4*b)).toInt=sequence b
  prompt : ∀ b,b<c.batches → (Snapshot.load32 m (B.base c.regions.prompt+4*b)).toInt=prompt b

theorem metadata (c : Host.Call) (B : Native.Bytes extent) (m : Physical.Memory)
    (start sequence prompt : Nat → Int) (guard : metadataChecks c start sequence prompt)
    (observed : MetadataObservation c B m start sequence prompt) : Snapshot.Metadata c B m := by
  refine ⟨?_,?_,?_⟩ <;> intro b hb
  · exact (observed.start b hb).trans (guard ⟨b,hb⟩).1
  · exact (observed.sequence b hb).trans (guard ⟨b,hb⟩).2.2
  · exact (observed.prompt b hb).trans (guard ⟨b,hb⟩).2.1

/-- Signed int32 metadata widths follow from their actual source words, not
from extra restrictions on already admitted payload values. -/
theorem abi (c : Host.Call) (v : Views) (B : Native.Bytes extent) (m : Physical.Memory)
    (guard : nativeChecks c v) (metadata : Snapshot.Metadata c B m) : Native.ABI c := by
  refine ⟨guard.1,guard.2.1,?_,?_⟩
  · intro b hb
    have h := BitVec.toInt_lt (x:=Snapshot.load32 m (B.base c.regions.start+4*b))
    rw [metadata.start b hb] at h
    norm_num at h
    exact_mod_cast h
  · intro b hb
    have h := BitVec.toInt_lt (x:=Snapshot.load32 m (B.base c.regions.sequence+4*b))
    rw [metadata.sequence b hb] at h
    norm_num at h
    exact_mod_cast h

def roleViews (c : Host.Call) (v : Views) : List (RegionName × View) :=
  [(c.regions.query,v.q),(c.regions.key,v.k),(c.regions.value,v.v),(c.regions.out,v.o),
   (c.regions.start,v.start),(c.regions.sequence,v.sequence),(c.regions.prompt,v.prompt)]

/-- The observed data pointers and available native word windows lie in the
same backing intervals checked by Python. These are Tensor storage facts. -/
structure StorageBinding (c : Host.Call) (v : Views) (B : Native.Bytes extent) : Prop where
  base : ∀ rg x,(rg,x)∈roleViews c v → B.base rg=x.metadata.dataPtr.toNat
  span : ∀ rg x,(rg,x)∈roleViews c v → 0<extent rg →
    x.storageBase ≤ B.base rg ∧ B.base rg+4*extent rg ≤ x.storageBase+x.storageBytes

theorem role_alignment (c : Host.Call) (v : Views) (B : Native.Bytes extent)
    (guard : tensorChecks c v) (storage : StorageBinding c v B) (rg : RegionName) (x : View)
    (role : (rg,x)∈roleViews c v) (nonempty : 0<x.metadata.numel) (off : Nat) :
    (B.base rg+4*off)%4=0 := by
  rcases guard with ⟨_,_,_,_,_,_,_,attrs,_⟩
  have hx : x∈all v := by
    simp only [roleViews,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at role
    rcases role with role|role|role|role|role|role|role <;> rcases role with ⟨_,rfl⟩ <;> simp [all]
  have alignment := (attrs x hx).2.2.2.2.2.resolve_left (by omega)
  rw [storage.base rg x role]
  simp [Nat.add_mod,alignment]

theorem needed_extents (c : Host.Call) (extent : RegionBounds) (h : Geometry.Checked c extent)
    (needed : Returned.Needed c) :
    ∀ rg,rg∈[c.regions.query,c.regions.key,c.regions.value,c.regions.out,
      c.regions.start,c.regions.sequence,c.regions.prompt] → 0<extent rg := by
  obtain ⟨b,hb,hl⟩ := needed
  have hd : 0<c.layout.D := by
    have width := h.width
    simp only [Finset.mem_insert,Finset.mem_singleton] at width
    change 0<c.args.k.dims
    omega
  have hh : 0<c.layout.heads := c.headsPositive
  have q := Geometry.query_bound c h b 0 0 0 hb hh hl hd
  have k := Geometry.key_bound c h b 0 0 0 hb hh (by omega) hd
  have v := Geometry.value_bound c h b 0 0 0 hb hh (by omega) hd
  have o := Geometry.output_bound c h b 0 0 0 hb hh hl hd
  intro rg hr
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl
  · omega
  · omega
  · omega
  · omega
  · have hs := h.start_span; omega
  · have hs := h.sequence_span; omega
  · have hs := h.prompt_span; omega

theorem separate_windows (a b : View) (abase bbase na nb : Nat)
    (sep : separate a b) (apos : 0<na) (bpos : 0<nb)
    (aspan : a.storageBase ≤ abase ∧ abase+4*na ≤ a.storageBase+a.storageBytes)
    (bspan : b.storageBase ≤ bbase ∧ bbase+4*nb ≤ b.storageBase+b.storageBytes) :
    abase+4*na ≤ bbase ∨ bbase+4*nb ≤ abase := by
  rcases sep with h|h|h|h <;> omega

theorem physical_separate (c : Host.Call) (v : Views) (B : Native.Bytes extent)
    (guard : tensorChecks c v) (geometry : Geometry.Checked c extent)
    (storage : StorageBinding c v B) (needed : Returned.Needed c) : Physical.Separate c B := by
  rcases guard with ⟨_,_,_,_,_,_,_,_,_,_,_,_,separated,_⟩
  have nonzero : c.batches≠0 := by obtain ⟨b,hb,_⟩:=needed;omega
  have sep := separated.resolve_left nonzero
  have positive := needed_extents c extent geometry needed
  have outpos : 0<extent c.regions.out := positive _ (by simp)
  have outspan := storage.span c.regions.out v.o (by simp [roleViews]) outpos
  have one (rg : RegionName) (x : View) (role : (rg,x)∈roleViews c v)
      (read : x∈reads v) (pos : 0<extent rg) :
      B.base c.regions.out+4*extent c.regions.out ≤ B.base rg ∨
        B.base rg+4*extent rg ≤ B.base c.regions.out :=
    separate_windows v.o x _ _ _ _ (sep x read) outpos pos outspan (storage.span rg x role pos)
  intro rg hr
  have readpos : 0<extent rg := positive rg (by
    simp only [Physical.reads,List.mem_cons,List.not_mem_nil,or_false] at hr ⊢
    tauto)
  simp only [Physical.reads,List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl|rfl
  · exact one _ v.q (by simp [roleViews]) (by simp [reads]) readpos
  · exact one _ v.k (by simp [roleViews]) (by simp [reads]) readpos
  · exact one _ v.v (by simp [roleViews]) (by simp [reads]) readpos
  · exact one _ v.start (by simp [roleViews]) (by simp [reads]) readpos
  · exact one _ v.sequence (by simp [roleViews]) (by simp [reads]) readpos
  · exact one _ v.prompt (by simp [roleViews]) (by simp [reads]) readpos

#axiomsClean metadata
#axiomsClean abi
#axiomsClean role_alignment
#axiomsClean physical_separate
end
end VeriTile.Bench.Repairs.ContextAttnFwd.Checked
