import bench.repairs.tritonbench_g.context_attn_llama.Proofs.Bytes
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.CheckedBridge

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Formula Program
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def roleViews (c : Call) (v : Views) : List (RegionName × View) :=
  [(c.regions.query,v.q),(c.regions.key,v.k),(c.regions.value,v.v),(c.regions.out,v.o),
    (c.regions.start,v.start),(c.regions.sequence,v.sequence),(c.regions.prompt,v.prompt),
    (c.regions.request,v.request),(c.regions.pages,v.pages)]

structure StorageBinding (c : Call) (v : Views) (B : Bytes.Windows extent) : Prop where
  base : ∀ rg x,(rg,x)∈roleViews c v → B.base rg=x.metadata.dataPtr.toNat
  span : ∀ rg x,(rg,x)∈roleViews c v → 0 < extent rg →
    x.storageBase≤B.base rg ∧ B.base rg+4*extent rg≤x.storageBase+x.storageBytes

theorem role_member (c : Call) (v : Views) (rg : RegionName) (x : View)
    (role : (rg,x)∈roleViews c v) : x∈all v := by
  simp only [roleViews,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at role
  rcases role with h|h|h|h|h|h|h|h|h <;> rcases h with ⟨_,rfl⟩ <;> simp [all]

theorem role_alignment (c : Call) (d : Dimensions) (v : Views) (B : Bytes.Windows extent)
    (tensors : tensorChecks c d v) (storage : StorageBinding c v B) (rg : RegionName) (x : View)
    (role : (rg,x)∈roleViews c v) (nonempty : 0 < x.metadata.numel) (off : Nat) :
    (B.base rg+4*off)%4=0 := by
  have align := (tensors.2.2.2.2.2.1 x (role_member c v rg x role)).2.2.2.2.2.2.resolve_left (by omega)
  rw [storage.base rg x role]
  simp [Nat.add_mod,align]

theorem source_strides (c : Call) (d : Dimensions) (v : Views) (extent : RegionBounds)
    (tensors : tensorChecks c d v) (geometry : geometryChecks c d extent) :
    v.q.strides=[c.layout.qRow,c.layout.qHead,c.layout.qDim] ∧
    v.k.strides=[c.layout.kRow,c.layout.kHead,c.layout.kDim] ∧
    v.v.strides=[c.layout.vRow,c.layout.vHead,c.layout.vDim] ∧
    v.o.strides=[c.layout.outRow,c.layout.outHead,c.layout.outDim] ∧
    v.pages.strides=[c.layout.reqRow,c.layout.reqToken] := by
  rcases geometry.1 with ⟨qr,qh,qd,orr,oh,od,kr,kh,kd,vr,vh,vd⟩
  simpa only [qr,qh,qd,orr,oh,od,kr,kh,kd,vr,vh,vd,geometry.2.1,geometry.2.2.1] using
    And.intro (tensors.1 v.q (by simp)).2.2.2
      (And.intro (tensors.2.1 v.k (by simp)).2.2.2
        (And.intro (tensors.2.1 v.v (by simp)).2.2.2
          (And.intro (tensors.1 v.o (by simp)).2.2.2 tensors.2.2.2.1.2.2.2)))

theorem needed_extents (c : Call) (d : Dimensions) (extent : RegionBounds)
    (g : geometryChecks c d extent) (needed : Bytes.Needed c) :
    0 < extent c.regions.out ∧ ∀ rg∈c.inputs,0 < extent rg := by
  obtain ⟨b,hb,hl⟩ := needed
  rcases g with ⟨strides,rr,rt,kv,group,grouped,capacity,packed,querySpace,cacheSpace,metaSpace,pageSpace⟩
  have cap := capacity ⟨b,hb⟩
  have qRows : 0 < d.queryRows := by have := cap.1; change c.start b+c.length b≤d.queryRows at this; omega
  have seq : 0 < c.length b+c.prompt b := by omega
  have cacheRows : 0 < d.cacheRows := Nat.zero_lt_of_lt (cap.2.2.2.2 ⟨0,seq⟩)
  have req : 0 < d.requests := Nat.zero_lt_of_lt cap.2.2.2.1
  have tokens : 0 < d.tokens := lt_of_lt_of_le seq cap.2.2.1
  have width : 0 < c.layout.D := by have h := c.channels; simp only [List.mem_cons,List.not_mem_nil,or_false] at h; omega
  have query : 0 < d.queryRows*c.layout.heads*c.layout.D := Nat.mul_pos (Nat.mul_pos qRows c.headsPositive) width
  have cache : 0 < d.cacheRows*c.kvHeads*c.layout.D := Nat.mul_pos (Nat.mul_pos cacheRows kv) width
  refine ⟨lt_of_lt_of_le query (querySpace _ (by simp)),?_⟩
  intro rg read
  simp only [Call.inputs,List.mem_cons,List.not_mem_nil,or_false] at read
  rcases read with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  · exact lt_of_lt_of_le query (querySpace _ (by simp))
  · exact lt_of_lt_of_le cache (cacheSpace _ (by simp))
  · exact lt_of_lt_of_le cache (cacheSpace _ (by simp))
  · exact lt_of_lt_of_le (by omega) (metaSpace _ (by simp))
  · exact lt_of_lt_of_le (by omega) (metaSpace _ (by simp))
  · exact lt_of_lt_of_le (by omega) (metaSpace _ (by simp))
  · exact lt_of_lt_of_le (by omega) (metaSpace _ (by simp))
  · exact lt_of_lt_of_le (Nat.mul_pos req tokens) pageSpace

theorem physical_separate (c : Call) (d : Dimensions) (v : Views) (B : Bytes.Windows extent)
    (tensors : tensorChecks c d v) (geometry : geometryChecks c d extent)
    (storage : StorageBinding c v B) (needed : Bytes.Needed c) : Bytes.Separate c B := by
  have nonzero : c.batches≠0 := by obtain ⟨b,hb,_⟩ := needed; omega
  have separated := tensors.2.2.2.2.2.2.resolve_left nonzero
  have positive := needed_extents c d extent geometry needed
  have outspan := storage.span c.regions.out v.o (by simp [roleViews]) positive.1
  have one (rg : RegionName) (x : View) (role : (rg,x)∈roleViews c v) (read : x∈reads v) (pos : 0 < extent rg) :
      B.base c.regions.out+4*extent c.regions.out≤B.base rg ∨ B.base rg+4*extent rg≤B.base c.regions.out :=
    VeriTile.Bench.Repairs.ContextAttnFwd.Checked.separate_windows v.o x _ _ _ _
      (separated x read) positive.1 pos outspan (storage.span rg x role pos)
  intro rg read
  have pos := positive.2 rg read
  simp only [Call.inputs,List.mem_cons,List.not_mem_nil,or_false] at read
  rcases read with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  · exact one _ v.q (by simp [roleViews]) (by simp [reads]) pos
  · exact one _ v.k (by simp [roleViews]) (by simp [reads]) pos
  · exact one _ v.v (by simp [roleViews]) (by simp [reads]) pos
  · exact one _ v.start (by simp [roleViews]) (by simp [reads]) pos
  · exact one _ v.sequence (by simp [roleViews]) (by simp [reads]) pos
  · exact one _ v.prompt (by simp [roleViews]) (by simp [reads]) pos
  · exact one _ v.request (by simp [roleViews]) (by simp [reads]) pos
  · exact one _ v.pages (by simp [roleViews]) (by simp [reads]) pos

#axiomsClean source_strides
#axiomsClean role_alignment
#axiomsClean physical_separate
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Checked
