import bench.tritonbench_g.context_attn_fwd.IO

namespace VeriTile.Bench.Repairs.ContextAttnFwd.RealLocality
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section

/-- Agreement only on the finite six typed input windows. -/
structure Agree (r : Regions) (extent : RegionBounds) (a b : Formula.Inputs) : Prop where
  query : ∀ off,off < extent r.query → a.q off=b.q off
  key : ∀ off,off < extent r.key → a.k off=b.k off
  value : ∀ off,off < extent r.value → a.v off=b.v off
  start : ∀ off,off < extent r.start → a.start off=b.start off
  sequence : ∀ off,off < extent r.sequence → a.sequence off=b.sequence off
  prompt : ∀ off,off < extent r.prompt → a.prompt off=b.prompt off

structure Metadata (p : Layout) (l : Formula.Launch) (a b : Formula.Inputs) : Prop where
  start : a.start (Formula.batch p l)=b.start (Formula.batch p l)
  sequence : a.sequence (Formula.batch p l)=b.sequence (Formula.batch p l)
  prompt : a.prompt (Formula.batch p l)=b.prompt (Formula.batch p l)

theorem metadata (p : Layout) (r : Regions) (extent : RegionBounds) (l : Formula.Launch)
    (a b : Formula.Inputs) (agreement : Agree r extent a b)
    (bounds : AddressBounds.Valid p r l a extent) : Metadata p l a b :=
  ⟨agreement.start _ bounds.start,agreement.sequence _ bounds.sequence,agreement.prompt _ bounds.prompt⟩

theorem Metadata.length (p : Layout) (l : Formula.Launch) (a b : Formula.Inputs)
    (h : Metadata p l a b) : Formula.sequence p l a=Formula.sequence p l b := by
  simp only [Formula.sequence,h.sequence,h.prompt]

theorem Metadata.limit (p : Layout) (l : Formula.Launch) (a b : Formula.Inputs)
    (h : Metadata p l a b) : Formula.limit p l a=Formula.limit p l b := by
  simp only [Formula.limit,h.sequence,h.prompt]

theorem Metadata.count (p : Layout) (l : Formula.Launch) (a b : Formula.Inputs)
    (h : Metadata p l a b) : Formula.count p l a=Formula.count p l b := by
  simp only [Formula.count,h.length p l a b,h.limit p l a b]

theorem Metadata.active (p : Layout) (l : Formula.Launch) (a b : Formula.Inputs)
    (h : Metadata p l a b) (i : RealIO.Lane p) : RealIO.active p l a i=RealIO.active p l b i := by
  simp only [RealIO.active,h.length p l a b]

theorem Metadata.address (p : Layout) (l : Formula.Launch) (a b : Formula.Inputs)
    (h : Metadata p l a b) (i : RealIO.Lane p) : RealIO.write p l a i=RealIO.write p l b i := by
  simp only [RealIO.write,AddressBounds.outputAddress,h.start]

theorem Metadata.bounds (p : Layout) (r : Regions) (l : Formula.Launch) (a b : Formula.Inputs)
    (h : Metadata p l a b) (extent : RegionBounds) (bounds : AddressBounds.Valid p r l a extent) :
    AddressBounds.Valid p r l b extent := by
  constructor
  · exact bounds.start
  · exact bounds.prompt
  · exact bounds.sequence
  · simpa only [←h.length p l a b,AddressBounds.queryAddress,←h.start] using bounds.query
  · simpa only [←h.count p l a b,←h.limit p l a b] using bounds.key
  · simpa only [←h.count p l a b,←h.limit p l a b] using bounds.value
  · simpa only [←h.length p l a b,AddressBounds.outputAddress,←h.start] using bounds.output

theorem key_bound (p : Layout) (r : Regions) (extent : RegionBounds) (l : Formula.Launch)
    (a : Formula.Inputs) (positive : 0 < p.BN) (bounds : AddressBounds.Valid p r l a extent)
    (j : Fin (Formula.count p l a*p.BN)) (d : Fin p.D) (active : (j.val : Int) < Formula.limit p l a) :
    Formula.batch p l*p.kBatch+j.val*p.kRow+(Formula.head p l/p.kvGroup)*p.kHead+d.val*p.kDim < extent r.key := by
  have hc : j.val/p.BN < Formula.count p l a := (Nat.div_lt_iff_lt_mul positive).mpr j.isLt
  have he : j.val/p.BN*p.BN+j.val%p.BN=j.val := by simpa only [Nat.mul_comm] using Nat.div_add_mod j.val p.BN
  have hb := bounds.key (j.val/p.BN) hc ⟨j.val%p.BN,Nat.mod_lt _ positive⟩ d (by simpa only [he] using active)
  have hn := AddressBounds.inside_toNat hb
  simpa only [AddressBounds.keyAddress,he,Nat.cast_add,Nat.cast_mul,Int.toNat_natCast] using hn

theorem value_bound (p : Layout) (r : Regions) (extent : RegionBounds) (l : Formula.Launch)
    (a : Formula.Inputs) (positive : 0 < p.BN) (bounds : AddressBounds.Valid p r l a extent)
    (j : Fin (Formula.count p l a*p.BN)) (d : Fin p.D) (active : (j.val : Int) < Formula.limit p l a) :
    Formula.batch p l*p.vBatch+j.val*p.vRow+(Formula.head p l/p.kvGroup)*p.vHead+d.val*p.vDim < extent r.value := by
  have hc : j.val/p.BN < Formula.count p l a := (Nat.div_lt_iff_lt_mul positive).mpr j.isLt
  have he : j.val/p.BN*p.BN+j.val%p.BN=j.val := by simpa only [Nat.mul_comm] using Nat.div_add_mod j.val p.BN
  have hb := bounds.value (j.val/p.BN) hc ⟨j.val%p.BN,Nat.mod_lt _ positive⟩ d (by simpa only [he] using active)
  have hn := AddressBounds.inside_toNat hb
  simpa only [AddressBounds.valueAddress,he,Nat.cast_add,Nat.cast_mul,Int.toNat_natCast] using hn

theorem query_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Formula.Launch)
    (a b : Formula.Inputs) (agreement : Agree r extent a b) (bounds : AddressBounds.Valid p r l a extent)
    (i : Fin p.BM) (d : Fin p.D) : Formula.query p l a i.val d.val=Formula.query p l b i.val d.val := by
  have hm := metadata p r extent l a b agreement bounds
  unfold Formula.query
  rw [←hm.length p l a b,←hm.start]
  split_ifs with h
  · exact agreement.query _ (AddressBounds.inside_toNat (bounds.query i d h))
  · rfl

theorem key_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Formula.Launch)
    (a b : Formula.Inputs) (positive : 0 < p.BN) (agreement : Agree r extent a b)
    (bounds : AddressBounds.Valid p r l a extent) (j : Fin (Formula.count p l a*p.BN)) (d : Fin p.D) :
    Formula.key p l a j.val d.val=Formula.key p l b j.val d.val := by
  have hm := metadata p r extent l a b agreement bounds
  unfold Formula.key
  rw [←hm.limit p l a b]
  split_ifs with h
  · exact agreement.key _ (key_bound p r extent l a positive bounds j d h)
  · rfl

theorem value_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Formula.Launch)
    (a b : Formula.Inputs) (positive : 0 < p.BN) (agreement : Agree r extent a b)
    (bounds : AddressBounds.Valid p r l a extent) (j : Fin (Formula.count p l a*p.BN)) (d : Fin p.D) :
    Formula.value p l a j.val d.val=Formula.value p l b j.val d.val := by
  have hm := metadata p r extent l a b agreement bounds
  unfold Formula.value
  rw [←hm.limit p l a b]
  split_ifs with h
  · exact agreement.value _ (value_bound p r extent l a positive bounds j d h)
  · rfl

theorem score_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Formula.Launch)
    (a b : Formula.Inputs) (positive : 0 < p.BN) (agreement : Agree r extent a b)
    (bounds : AddressBounds.Valid p r l a extent) (i : Fin p.BM) (j : Fin (Formula.count p l a*p.BN))
    (scale : ℝ) : Formula.score p l a scale i.val j.val=Formula.score p l b scale i.val j.val := by
  have hm := metadata p r extent l a b agreement bounds
  unfold Formula.score
  rw [←hm.prompt]
  split_ifs
  · congr 1
    apply Finset.sum_congr rfl
    intro d _
    rw [query_eq p r extent l a b agreement bounds i d,key_eq p r extent l a b positive agreement bounds j d]
  · rfl

theorem expected_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Formula.Launch)
    (a b : Formula.Inputs) (positive : 0 < p.BN) (agreement : Agree r extent a b)
    (bounds : AddressBounds.Valid p r l a extent) (scale : ℝ) (i : RealIO.Lane p) :
    RealIO.closedExpected p scale l a i=RealIO.closedExpected p scale l b i := by
  have hm := metadata p r extent l a b agreement bounds
  unfold RealIO.closedExpected
  rw [←hm.count p l a b]
  congr 1
  · apply Finset.sum_congr rfl
    intro j _
    rw [score_eq p r extent l a b positive agreement bounds i.1 j scale,
      value_eq p r extent l a b positive agreement bounds j i.2.1]
  · apply Finset.sum_congr rfl
    intro j _
    rw [score_eq p r extent l a b positive agreement bounds i.1 j scale]

#axiomsClean expected_eq
end
end VeriTile.Bench.Repairs.ContextAttnFwd.RealLocality
