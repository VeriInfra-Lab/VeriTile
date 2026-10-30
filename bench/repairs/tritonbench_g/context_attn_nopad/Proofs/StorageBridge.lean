import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Bytes
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.CheckedBridge

namespace VeriTile.Bench.Repairs.ContextAttnNopad.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def roleViews (r : Regions) (v : Views) : List (RegionName × View) :=
  [(r.query,v.q),(r.key,v.k),(r.value,v.v),(r.out,v.o),(r.start,v.start),(r.sequence,v.length)]

structure StorageBinding (r : Regions) (v : Views) (B : Bytes.Windows extent) : Prop where
  base : ∀ rg x,(rg,x)∈roleViews r v → B.base rg=x.metadata.dataPtr.toNat
  span : ∀ rg x,(rg,x)∈roleViews r v → 0 < extent rg →
    x.storageBase≤B.base rg ∧ B.base rg+4*extent rg≤x.storageBase+x.storageBytes

theorem source_strides (c : Call) (r : Regions) (v : Views) (extent : RegionBounds)
    (tensors : tensorChecks c v) (geometry : geometryChecks c r extent) :
    v.q.strides=[c.q.rowStride,c.q.headStride,c.q.dimStride] ∧
    v.k.strides=[c.k.rowStride,c.k.headStride,c.k.dimStride] ∧
    v.v.strides=[c.v.rowStride,c.v.headStride,c.v.dimStride] ∧
    v.o.strides=[c.o.rowStride,c.o.headStride,c.o.dimStride] := by
  rcases geometry with ⟨_,_,k,value,o,contiguous,_⟩
  have q := (tensors.1 v.q (by simp)).2.2.2
  have kv := (tensors.1 v.k (by simp)).2.2.2
  have vv := (tensors.1 v.v (by simp)).2.2.2
  have ov := (tensors.1 v.o (by simp)).2.2.2
  simpa only [k,value,o,contiguous.1,contiguous.2.1,contiguous.2.2] using And.intro q (And.intro kv (And.intro vv ov))

theorem role_alignment (c : Call) (r : Regions) (v : Views) (B : Bytes.Windows extent)
    (guard : tensorChecks c v) (storage : StorageBinding r v B) (rg : RegionName) (x : View)
    (role : (rg,x)∈roleViews r v) (nonempty : 0 < x.metadata.numel) (off : Nat) :
    (B.base rg+4*off)%4=0 := by
  have member : x∈all v := by
    simp only [roleViews,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at role
    rcases role with h|h|h|h|h|h <;> rcases h with ⟨_,rfl⟩ <;> simp [all]
  have alignment := (guard.2.2.2.1 x member).2.2.2.2.2.2.resolve_left (by omega)
  rw [storage.base rg x role]
  simp [Nat.add_mod,alignment]

theorem needed_extents (c : Call) (r : Regions) (extent : RegionBounds)
    (g : geometryChecks c r extent) (needed : Bytes.Needed c) :
    ∀ rg,rg∈[r.query,r.key,r.value,r.out,r.start,r.sequence] → 0 < extent rg := by
  obtain ⟨b,hb,hl⟩ := needed
  rcases g with ⟨width,heads,k,value,o,contiguous,capacity,packed,space,se,le⟩
  have rows := (capacity ⟨b,hb⟩).1
  change c.start b+c.length b≤c.q.rows at rows
  have dim : 0 < c.q.dims := by
    simp only [Finset.mem_insert,Finset.mem_singleton] at width
    omega
  have payload : 0 < c.q.rows*c.q.heads*c.q.dims := Nat.mul_pos (Nat.mul_pos (by omega) heads) dim
  intro rg member
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl
  · exact lt_of_lt_of_le payload (space r.query (by simp))
  · exact lt_of_lt_of_le payload (space r.key (by simp))
  · exact lt_of_lt_of_le payload (space r.value (by simp))
  · exact lt_of_lt_of_le payload (space r.out (by simp))
  · omega
  · omega

theorem physical_separate (c : Call) (r : Regions) (v : Views) (B : Bytes.Windows extent)
    (guard : tensorChecks c v) (geometry : geometryChecks c r extent)
    (storage : StorageBinding r v B) (needed : Bytes.Needed c) : Bytes.Separate c r B := by
  have nonzero : c.batches≠0 := by obtain ⟨b,hb,_⟩ := needed; omega
  have separated := guard.2.2.2.2.resolve_left nonzero
  have positive := needed_extents c r extent geometry needed
  have outpos : 0 < extent r.out := positive _ (by simp)
  have outspan := storage.span r.out v.o (by simp [roleViews]) outpos
  have one (rg : RegionName) (x : View) (role : (rg,x)∈roleViews r v) (read : x∈reads v)
      (pos : 0 < extent rg) :
      B.base r.out+4*extent r.out≤B.base rg ∨ B.base rg+4*extent rg≤B.base r.out :=
    VeriTile.Bench.Repairs.ContextAttnFwd.Checked.separate_windows v.o x _ _ _ _
      (separated x read) outpos pos outspan (storage.span rg x role pos)
  intro rg read
  have pos : 0 < extent rg := positive rg (by
    simp only [Bytes.readRegions,List.mem_cons,List.not_mem_nil,or_false] at read ⊢
    tauto)
  simp only [Bytes.readRegions,List.mem_cons,List.not_mem_nil,or_false] at read
  rcases read with rfl|rfl|rfl|rfl|rfl
  · exact one _ v.q (by simp [roleViews]) (by simp [reads]) pos
  · exact one _ v.k (by simp [roleViews]) (by simp [reads]) pos
  · exact one _ v.v (by simp [roleViews]) (by simp [reads]) pos
  · exact one _ v.start (by simp [roleViews]) (by simp [reads]) pos
  · exact one _ v.length (by simp [roleViews]) (by simp [reads]) pos

#axiomsClean source_strides
#axiomsClean role_alignment
#axiomsClean physical_separate
end
end VeriTile.Bench.Repairs.ContextAttnNopad.Checked
