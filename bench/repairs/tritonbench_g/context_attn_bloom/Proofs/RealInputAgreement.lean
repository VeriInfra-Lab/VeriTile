import bench.repairs.tritonbench_g.context_attn_bloom.IO

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealInputAgreement
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 20000

/-- Finite input equality at declared capacities, with only the current
program's scalar metadata. It says nothing about execution or output values. -/
structure Agree (r : Regions) (extent : RegionBounds) (l : Launch) (a b : Math.Inputs) : Prop where
  start : a.start l.batch=b.start l.batch
  sequence : a.sequence l.batch=b.sequence l.batch
  prompt : a.prompt l.batch=b.prompt l.batch
  request : a.request l.batch=b.request l.batch
  query : ∀ off,off<extent r.query → a.q off=b.q off
  key : ∀ off,off<extent r.key → a.k off=b.k off
  value : ∀ off,off<extent r.value → a.v off=b.v off
  tokens : ∀ off,off<extent r.tokens → a.tokens off=b.tokens off

theorem Agree.sequence_eq (h : Agree r extent l a b) : Math.sequence a l=Math.sequence b l := by
  simp only [Math.sequence,h.sequence,h.prompt]
theorem Agree.limit_eq (h : Agree r extent l a b) : Math.limit p l a=Math.limit p l b := by
  simp only [Math.limit,h.sequence,h.prompt]
theorem Agree.count_eq (h : Agree r extent l a b) : Math.count p l a=Math.count p l b := by
  simp only [Math.count,h.sequence_eq,h.limit_eq]

theorem token_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Launch) (a b : Math.Inputs)
    (h : Agree r extent l a b) (hb : AddressBounds.Valid p r l b extent) (hBN : 0<p.BN)
    (j : Nat) (visited : j<Math.count p l b*p.BN) (active : (j:Int)<Math.limit p l b) :
    Math.token p l a j=Math.token p l b j := by
  have hc : j/p.BN<Math.count p l b := (Nat.div_lt_iff_lt_mul hBN).mpr visited
  have equal : j/p.BN*p.BN+j%p.BN=j := by simpa only [Nat.mul_comm] using Nat.div_add_mod j p.BN
  have backed := hb.tokens (j/p.BN) hc ⟨j%p.BN,Nat.mod_lt j hBN⟩ (by simpa only [equal] using active)
  have bound := AddressBounds.inside_toNat backed
  simp only [AddressBounds.tokenAddress,equal] at bound
  simp only [Math.token,h.request]
  apply h.tokens
  simpa only [Int.mul_comm] using bound

theorem query_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Launch) (a b : Math.Inputs)
    (h : Agree r extent l a b) (hb : AddressBounds.Valid p r l b extent)
    (i : Fin p.BM) (d : Fin p.D) : Math.query p l a i.val d.val=Math.query p l b i.val d.val := by
  simp only [Math.query,h.sequence_eq,h.start]
  by_cases active : (Math.queryRow p l i.val : Int)<Math.sequence b l ∧ d.val<p.headDim
  · rw [if_pos active,if_pos active]
    exact h.query _ (AddressBounds.inside_toNat (hb.query i d active.1 active.2))
  · rw [if_neg active,if_neg active]

theorem key_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Launch) (a b : Math.Inputs)
    (h : Agree r extent l a b) (hb : AddressBounds.Valid p r l b extent) (hBN : 0<p.BN)
    (j : Nat) (visited : j<Math.count p l b*p.BN) (d : Fin p.D) :
    Math.key p l a j d.val=Math.key p l b j d.val := by
  simp only [Math.key,h.limit_eq]
  by_cases active : (j:Int)<Math.limit p l b ∧ d.val<p.headDim
  · rw [if_pos active,if_pos active,token_eq p r extent l a b h hb hBN j visited active.1]
    have hc : j/p.BN<Math.count p l b := (Nat.div_lt_iff_lt_mul hBN).mpr visited
    have equal : j/p.BN*p.BN+j%p.BN=j := by simpa only [Nat.mul_comm] using Nat.div_add_mod j p.BN
    have backed := hb.key (j/p.BN) hc ⟨j%p.BN,Nat.mod_lt j hBN⟩ d (by simpa only [equal] using active.1) active.2
    apply h.key
    simpa only [equal] using AddressBounds.inside_toNat backed
  · rw [if_neg active,if_neg active]

theorem value_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Launch) (a b : Math.Inputs)
    (h : Agree r extent l a b) (hb : AddressBounds.Valid p r l b extent) (hBN : 0<p.BN)
    (j : Nat) (visited : j<Math.count p l b*p.BN) (d : Fin p.D) :
    Math.value p l a j d.val=Math.value p l b j d.val := by
  simp only [Math.value,h.limit_eq]
  by_cases active : (j:Int)<Math.limit p l b ∧ d.val<p.headDim
  · rw [if_pos active,if_pos active,token_eq p r extent l a b h hb hBN j visited active.1]
    have hc : j/p.BN<Math.count p l b := (Nat.div_lt_iff_lt_mul hBN).mpr visited
    have equal : j/p.BN*p.BN+j%p.BN=j := by simpa only [Nat.mul_comm] using Nat.div_add_mod j p.BN
    have backed := hb.value (j/p.BN) hc ⟨j%p.BN,Nat.mod_lt j hBN⟩ d (by simpa only [equal] using active.1) active.2
    apply h.value
    simpa only [equal] using AddressBounds.inside_toNat backed
  · rw [if_neg active,if_neg active]

theorem score_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Launch) (a b : Math.Inputs)
    (h : Agree r extent l a b) (hb : AddressBounds.Valid p r l b extent) (hBN : 0<p.BN)
    (scale : ℝ) (i : Fin p.BM) (j : Nat) (visited : j<Math.count p l b*p.BN) :
    Math.score p l a scale i.val j=Math.score p l b scale i.val j := by
  simp only [Math.score,h.prompt]
  congr 2
  apply Finset.sum_congr rfl
  intro d _
  rw [query_eq p r extent l a b h hb i d,key_eq p r extent l a b h hb hBN j visited d]

theorem denominator_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Launch) (a b : Math.Inputs)
    (h : Agree r extent l a b) (hb : AddressBounds.Valid p r l b extent) (hBN : 0<p.BN)
    (scale : ℝ) (i : Fin p.BM) (c : Nat) :
    Formula.denominator p l a scale i.val c=Formula.denominator p l b scale i.val c := by
  unfold Formula.denominator
  rw [h.count_eq (p:=p)]
  apply Finset.sum_congr rfl
  intro j _
  rw [score_eq p r extent l a b h hb hBN scale i j.val j.isLt]

theorem numerator_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Launch) (a b : Math.Inputs)
    (h : Agree r extent l a b) (hb : AddressBounds.Valid p r l b extent) (hBN : 0<p.BN)
    (scale : ℝ) (i : Fin p.BM) (d : Fin p.D) (c : Nat) :
    Formula.numerator p l a scale i.val d.val c=Formula.numerator p l b scale i.val d.val c := by
  unfold Formula.numerator
  rw [h.count_eq (p:=p)]
  apply Finset.sum_congr rfl
  intro j _
  rw [score_eq p r extent l a b h hb hBN scale i j.val j.isLt,
    value_eq p r extent l a b h hb hBN j.val j.isLt d]

theorem expected_eq (p : Layout) (r : Regions) (extent : RegionBounds) (l : Launch) (a b : Math.Inputs)
    (h : Agree r extent l a b) (hb : AddressBounds.Valid p r l b extent) (hBN : 0<p.BN)
    (scale : ℝ) (i : Fin p.BM) (d : Fin p.D) (c : Nat) :
    Formula.expected p l a scale i.val d.val c=Formula.expected p l b scale i.val d.val c := by
  induction c with
  | zero=>rfl
  | succ c ih=>
    simp only [Formula.expected,ih,h.prompt,
      denominator_eq p r extent l a b h hb hBN scale i,
      numerator_eq p r extent l a b h hb hBN scale i d]

#axiomsClean expected_eq
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealInputAgreement
