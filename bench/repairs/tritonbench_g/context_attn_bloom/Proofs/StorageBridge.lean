import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.Bytes
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.CheckedBridge

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open FPSuccess (Call)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def roleViews (r : Regions) (v : Views) : List (RegionName × View) :=
  [(r.query,v.q),(r.key,v.k),(r.value,v.v),(r.out,v.o),(r.start,v.start),(r.sequence,v.sequence),
    (r.prompt,v.prompt),(r.request,v.request),(r.tokens,v.pages)]

structure StorageBinding (r : Regions) (v : Views) (B : Bytes.Windows extent) : Prop where
  base : ∀ rg x,(rg,x)∈roleViews r v → B.base rg=x.metadata.dataPtr.toNat
  span : ∀ rg x,(rg,x)∈roleViews r v → 0 < extent rg →
    x.storageBase ≤ B.base rg ∧ B.base rg+4*extent rg ≤ x.storageBase+x.storageBytes

theorem role_member (r : Regions) (v : Views) (rg : RegionName) (x : View)
    (role : (rg,x)∈roleViews r v) : x∈all v := by
  simp only [roleViews,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at role
  rcases role with h|h|h|h|h|h|h|h|h <;> rcases h with ⟨_,rfl⟩ <;> simp [all]

theorem role_alignment (a : Host.Arguments) (r : Regions) (v : Views) (B : Bytes.Windows extent)
    (tensors : tensorChecks a v) (storage : StorageBinding r v B) (rg : RegionName) (x : View)
    (role : (rg,x)∈roleViews r v) (nonempty : 0 < x.metadata.numel) (off : Nat) :
    (B.base rg+4*off)%4=0 := by
  have align := (tensors.2.2.2.2.1 x (role_member r v rg x role)).2.2.2.2.2.2.resolve_left (by omega)
  rw [storage.base rg x role]
  simp [Nat.add_mod,align]

theorem source_strides (a : Host.Arguments) (r : Regions) (v : Views) (extent : RegionBounds)
    (tensors : tensorChecks a v) (geometry : geometryChecks a r extent) :
    v.q.strides=[(Host.layout a).qRow,(Host.layout a).qHead,(Host.layout a).qDim] ∧
    v.k.strides=[(Host.layout a).kRow,(Host.layout a).kHead,(Host.layout a).kDim] ∧
    v.v.strides=[(Host.layout a).vRow,(Host.layout a).vHead,(Host.layout a).vDim] ∧
    v.o.strides=[(Host.layout a).outRow,(Host.layout a).outHead,(Host.layout a).outDim] ∧
    v.pages.strides=[(Host.layout a).requestRow,(Host.layout a).requestToken] := by
  refine ⟨(tensors.1 (v.q,a.q) (by simp)).2.1,(tensors.1 (v.k,a.k) (by simp)).2.1,
    (tensors.1 (v.v,a.v) (by simp)).2.1,(tensors.1 (v.o,a.o) (by simp)).2.1,?_⟩
  simpa only [Host.layout,geometry.1.2.2.2.2.1,geometry.1.2.2.2.2.2] using tensors.2.2.1.2.2.2

theorem needed_extents (c : Call) (extent : RegionBounds)
    (g : geometryChecks c.args c.regions extent) (m : metadataChecks c.args c.input) (needed : Bytes.Needed c) :
    0 < extent c.regions.out ∧ ∀ rg∈readRegions c.regions,0 < extent rg := by
  obtain ⟨b,hb,hl⟩ := needed
  rcases g with ⟨strides,hq,hk,grouped,qk,kv,ok,oh,vh,orows,vrows,dlo,dhi,
    qspace,kspace,vspace,ospace,metaspace,tokenspace⟩
  have md : 0 ≤ c.input.start b ∧ 0 ≤ c.input.prompt b ∧ c.input.prompt b ≤ c.input.sequence b ∧
      c.input.sequence b-c.input.prompt b ≤ c.args.maxInput ∧
      c.input.start b+(c.input.sequence b-c.input.prompt b) ≤ c.args.q.rows ∧
      c.input.sequence b ≤ c.args.requestColumns ∧ 0 ≤ c.input.request b ∧ c.input.request b < c.args.requestRows ∧
      ∀ j : Fin (c.input.sequence b).toNat,
        0 ≤ c.input.tokens (c.args.requestColumns*(c.input.request b).toNat+j) ∧
        c.input.tokens (c.args.requestColumns*(c.input.request b).toNat+j) < c.args.k.rows := m.1 ⟨b,hb⟩
  have qRows : 0 < c.args.q.rows := by have := md.2.2.2.2.1; omega
  have seq : 0 < (c.input.sequence b).toNat := by have := md.2.1; omega
  have page := md.2.2.2.2.2.2.2.2 ⟨0,seq⟩
  have cacheRows : 0 < c.args.k.rows := by omega
  have req : 0 < c.args.requestRows := by have := md.2.2.2.2.2.2.1; have := md.2.2.2.2.2.2.2.1; omega
  have tokens : 0 < c.args.requestColumns := by have := md.2.2.2.2.2.1; omega
  have width : 0 < c.args.q.dims := by omega
  have query : 0 < c.args.q.rows*c.args.q.heads*c.args.q.dims := Nat.mul_pos (Nat.mul_pos qRows hq) width
  have cache : 0 < c.args.k.rows*c.args.k.heads*c.args.k.dims := Nat.mul_pos (Nat.mul_pos cacheRows hk) (by omega)
  have output : 0 < extent c.regions.out := by rw [orows,oh,ok,←qk] at ospace; omega
  refine ⟨output,?_⟩
  intro rg read
  simp only [readRegions,List.mem_cons,List.not_mem_nil,or_false] at read
  rcases read with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  · omega
  · omega
  · rw [vrows,vh,←kv] at vspace; omega
  · exact lt_of_lt_of_le (by omega) (metaspace _ (by simp))
  · exact lt_of_lt_of_le (by omega) (metaspace _ (by simp))
  · exact lt_of_lt_of_le (by omega) (metaspace _ (by simp))
  · exact lt_of_lt_of_le (by omega) (metaspace _ (by simp))
  · exact lt_of_lt_of_le (Nat.mul_pos req tokens) tokenspace

theorem physical_separate (c : Call) (v : Views) (B : Bytes.Windows extent)
    (tensors : tensorChecks c.args v) (geometry : geometryChecks c.args c.regions extent)
    (metadata : metadataChecks c.args c.input) (storage : StorageBinding c.regions v B)
    (needed : Bytes.Needed c) : Bytes.Separate c B := by
  have nonzero : c.args.batch≠0 := by obtain ⟨b,hb,_⟩ := needed; omega
  have separated := tensors.2.2.2.2.2.resolve_left nonzero
  have positive := needed_extents c extent geometry metadata needed
  have outspan := storage.span c.regions.out v.o (by simp [roleViews]) positive.1
  have one (rg : RegionName) (x : View) (role : (rg,x)∈roleViews c.regions v) (read : x∈reads v) (pos : 0 < extent rg) :
      B.base c.regions.out+4*extent c.regions.out ≤ B.base rg ∨ B.base rg+4*extent rg ≤ B.base c.regions.out :=
    VeriTile.Bench.Repairs.ContextAttnFwd.Checked.separate_windows v.o x _ _ _ _
      (separated x read) positive.1 pos outspan (storage.span rg x role pos)
  intro rg read
  have pos := positive.2 rg read
  simp only [readRegions,List.mem_cons,List.not_mem_nil,or_false] at read
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
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Checked
