import bench.tritonbench_g.chunk_gated_attention.Sources.StagedLayout
import VeriTile.Triton.Launch.Composition

/-! Ownership of each actual history/final cell, across all flattened programs
and batches. These are derived integer/address facts, not launch assumptions. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.Ownership
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram StagedLayout
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

theorem state_owner (p : Layout) (l m : Launch) (e f : Fin p.BK) (i j : Fin p.BV)
    (he : RealHStore.active p l e.val i.val) (hf : RealHStore.active p m f.val j.val)
    (h : stateAddress p l (keyIndex p l e.val) (valueIndex p l i.val) =
      stateAddress p m (keyIndex p m f.val) (valueIndex p m j.val)) : l=m := by
  have h1 := split_address he.2 hf.2 (show
    (l.head*p.K+keyIndex p l e.val)*p.V+valueIndex p l i.val =
      (m.head*p.K+keyIndex p m f.val)*p.V+valueIndex p m j.val by
        simpa [stateAddress,Nat.add_mul,Nat.mul_assoc] using h)
  have h2 := split_address he.1 hf.1 h1.1
  have hk := split_address e.isLt f.isLt h2.2
  have hv := split_address i.isLt j.isLt h1.2
  cases l; cases m
  simp_all

theorem history_owner (p : Layout) (l m : Launch) (hp : Canonical p)
    (c d : Nat) (e f : Fin p.BK) (i j : Fin p.BV)
    (hc : c < p.NT) (hd : d < p.NT)
    (he : RealHStore.active p l e.val i.val) (hf : RealHStore.active p m f.val j.val)
    (h : historyAddress p l c (keyIndex p l e.val) (valueIndex p l i.val) =
      historyAddress p m d (keyIndex p m f.val) (valueIndex p m j.val)) : l=m := by
  have h1 := split_address he.2 hf.2 (show
    ((l.head*p.NT+c)*p.K+keyIndex p l e.val)*p.V+valueIndex p l i.val =
      ((m.head*p.NT+d)*p.K+keyIndex p m f.val)*p.V+valueIndex p m j.val by
        simpa [historyAddress,hp.historyHead,hp.historyKey,hp.historyValue,Nat.add_mul,Nat.mul_assoc] using h)
  have h2 := split_address he.1 hf.1 h1.1
  have h3 := split_address hc hd h2.1
  have hk := split_address e.isLt f.isLt h2.2
  have hv := split_address i.isLt j.isLt h1.2
  cases l; cases m
  simp_all

theorem region_owner (p : Layout) (l m : Launch) (r : Regions) (storeFinal : Bool)
    (hp : Canonical p) (hr : storeFinal = Bool.true → r.history ≠ r.final)
    (cell : MemCellAddr)
    (hl : RebasedExecution.footprint p l r storeFinal cell)
    (hm : RebasedExecution.footprint p m r storeFinal cell) : l=m := by
  rcases hl with hl|hl <;> rcases hm with hm|hm
  · rcases hl with ⟨hl,c,e,i,hc,he,ho⟩
    rcases hm with ⟨hm,d,f,j,hd,hf,hp'⟩
    apply history_owner p l m hp c d e f i j hc hd he hf
    exact ho.symm.trans hp'
  · exact False.elim (hr hm.1 (hl.1.symm.trans hm.2.1))
  · exact False.elim (hr hl.1 (hm.1.symm.trans hl.2.1))
  · rcases hl with ⟨_,hl,e,i,he,ho⟩
    rcases hm with ⟨_,hm,f,j,hf,hp'⟩
    apply state_owner p l m e f i j he hf
    simpa [RealHStore.address,RealHFinal.base,stateAddress,Nat.mul_one] using ho.symm.trans hp'

theorem region_bound (p : Layout) (heads : Nat) (l : Launch) (r : Regions)
    (gateK hasSeed storeFinal : Bool) (hp : Canonical p) (hl : TileValid p heads l)
    (b : RegionBounds) (ha : Allocated p heads r gateK hasSeed storeFinal b)
    (rg : RegionName) (off : Nat) (hw : RebasedExecution.footprint p l r storeFinal (rg,off)) : off < b rg := by
  have hb := bounds p heads l r gateK hasSeed storeFinal hp hl b ha
  rcases hw with hw|hw
  · rcases hw with ⟨rfl,c,e,j,hc,he,rfl⟩
    have h := (hb.loop c hc).history e j (by simpa using And.intro e.isLt j.isLt)
    convert h using 1 <;> simp only [RealHLoop.address,RealHStore.address,RealHStep.historyBase,keyIndex,valueIndex,Nat.zero_add] <;> ring
  · rcases hw with ⟨hon,rfl,e,j,he,rfl⟩
    have h := hb.final hon e j (by simpa using And.intro e.isLt j.isLt)
    convert h using 1 <;> simp only [RealHStore.address,RealHFinal.base,keyIndex,valueIndex,Nat.zero_add,Nat.mul_one] <;> ring

theorem flat_owner (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (heads : Nat) (l m : Launch) (r : Regions) (gateK hasSeed storeFinal : Bool)
    (hp : Canonical p) (hl : TileValid p heads l) (hm : TileValid p heads m)
    (ha : Allocated p heads r gateK hasSeed storeFinal A.extent)
    (hr : storeFinal = Bool.true → r.history ≠ r.final) (cell : MemCellAddr)
    (hwl : RebasedFlat.footprint A p l r storeFinal cell)
    (hwm : RebasedFlat.footprint A p m r storeFinal cell) : l=m := by
  rcases hwl with ⟨_,rl,ol,hwl,hal⟩
  rcases hwm with ⟨_,rm,om,hwm,ham⟩
  have hbl := region_bound p heads l r gateK hasSeed storeFinal hp hl A.extent ha rl ol hwl
  have hbm := region_bound p heads m r gateK hasSeed storeFinal hp hm A.extent ha rm om hwm
  have hrl : rl ∈ A.regions := by by_contra hn; rw [hcov rl hn] at hbl; omega
  have hrm : rm ∈ A.regions := by by_contra hn; rw [hcov rm hn] at hbm; omega
  have heq := congrArg A.decode (hal.trans ham.symm)
  rw [A.decode_addr hd hrl hbl,A.decode_addr hd hrm hbm] at heq
  have heq' : (rl,ol)=(rm,om) := Option.some.inj heq
  cases heq'
  exact region_owner p l m r storeFinal hp hr (rl,ol) hwl hwm

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.Ownership
