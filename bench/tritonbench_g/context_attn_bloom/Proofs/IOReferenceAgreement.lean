import bench.tritonbench_g.context_attn_bloom.Proofs.IOSourceSafety

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.IOReferenceAgreement
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
open FPSuccess FPReference
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]

/-- Equality only at declared finite input cells. No execution or final
output equality is assumed, and arbitrary tags/default reads remain admitted. -/
structure Agree (r : Regions) (extent : RegionBounds) (a b : FPReference.Inputs α) : Prop where
  batch : a.batch=b.batch
  head : a.head=b.head
  kvHead : a.kvHead=b.kvHead
  block : a.block=b.block
  start : a.start=b.start
  length : a.length=b.length
  prompt : a.prompt=b.prompt
  request : a.request=b.request
  query : ∀ off,off<extent r.query → a.query off=b.query off
  key : ∀ off,off<extent r.key → a.key off=b.key off
  value : ∀ off,off<extent r.value → a.value off=b.value off
  pages : ∀ off,off<extent r.tokens → a.pages off=b.pages off

theorem Agree.row_eq {p : Layout} (h : Agree r extent a b) : FPReference.row p a=FPReference.row p b := by
  funext i
  simp only [FPReference.row,h.block]
theorem Agree.end_eq {p : Layout} (h : Agree r extent a b) : keyEnd p a=keyEnd p b := by
  simp only [keyEnd,h.block,h.length,h.prompt]
theorem Agree.mask_eq {p : Layout} (h : Agree r extent a b) : blockMask p a=blockMask p b := by
  simp only [blockMask,h.block,h.length]

theorem query_eq (p : Layout) (r : Regions) (extent : RegionBounds) (M : Algebra α)
    (a b : FPReference.Inputs α) (h : Agree r extent a b) (hb : IOSourceSafety.Bounds p r b extent) :
    queryTile p M a=queryTile p M b := by
  funext i
  simp only [queryTile,h.row_eq,h.length,h.start,h.head]
  by_cases active : (FPReference.row p b i.1.val : Int)<b.length ∧ i.2.1.val<p.headDim
  · simp only [active,if_true]
    rw [h.query _ (hb.query i.1 i.2.1 active.1 active.2)]
  · simp only [active,if_false]

theorem page_eq (p : Layout) (r : Regions) (extent : RegionBounds)
    (a b : FPReference.Inputs α) (h : Agree r extent a b) (hb : IOSourceSafety.Bounds p r b extent)
    (cur : Int) (nonnegative : 0≤cur) (aligned : ∃ n : Nat,cur=(n*p.BN : Nat))
    (live : cur<(blockMask p b : Int)*keyEnd p b) (j : Fin p.BN) :
    page p a cur j.val=page p b cur j.val := by
  simp only [page,h.end_eq,h.request]
  by_cases active : cur+j.val<keyEnd p b
  · simp only [active,if_true]
    exact h.pages _ (hb.tokens cur nonnegative aligned live j active)
  · simp only [active,if_false]

theorem key_eq (p : Layout) (r : Regions) (extent : RegionBounds) (M : Algebra α)
    (a b : FPReference.Inputs α) (h : Agree r extent a b) (hb : IOSourceSafety.Bounds p r b extent)
    (cur : Int) (nonnegative : 0≤cur) (aligned : ∃ n : Nat,cur=(n*p.BN : Nat))
    (live : cur<(blockMask p b : Int)*keyEnd p b) : keyTile p M a cur=keyTile p M b cur := by
  funext i
  simp only [keyTile,h.end_eq,h.kvHead,page_eq p r extent a b h hb cur nonnegative aligned live i.2.1]
  by_cases active : cur+i.2.1.val<keyEnd p b ∧ i.1.val<p.headDim
  · simp only [active,if_true]
    rw [h.key _ (hb.key cur nonnegative aligned live i.2.1 i.1 active.1 active.2)]
  · simp only [active,if_false]

theorem value_eq (p : Layout) (r : Regions) (extent : RegionBounds) (M : Algebra α)
    (a b : FPReference.Inputs α) (h : Agree r extent a b) (hb : IOSourceSafety.Bounds p r b extent)
    (cur : Int) (nonnegative : 0≤cur) (aligned : ∃ n : Nat,cur=(n*p.BN : Nat))
    (live : cur<(blockMask p b : Int)*keyEnd p b) : valueTile p M a cur=valueTile p M b cur := by
  funext i
  simp only [valueTile,h.end_eq,h.kvHead,page_eq p r extent a b h hb cur nonnegative aligned live i.1]
  by_cases active : cur+i.1.val<keyEnd p b ∧ i.2.1.val<p.headDim
  · simp only [active,if_true]
    rw [h.value _ (hb.value cur nonnegative aligned live i.1 i.2.1 active.1 active.2)]
  · simp only [active,if_false]

theorem advance_eq (p : Layout) (r : Regions) (extent : RegionBounds) (M : Algebra α) (dots : Dot α)
    (scale : ℝ) (a b : FPReference.Inputs α) (h : Agree r extent a b) (hb : IOSourceSafety.Bounds p r b extent)
    (cur : Int) (nonnegative : 0≤cur) (aligned : ∃ n : Nat,cur=(n*p.BN : Nat))
    (live : cur<(blockMask p b : Int)*keyEnd p b) (old : Acc p α) :
    advance p M dots scale a cur old=advance p M dots scale b cur old := by
  simp only [advance,query_eq p r extent M a b h hb,
    key_eq p r extent M a b h hb cur nonnegative aligned live,
    value_eq p r extent M a b h hb cur nonnegative aligned live,h.row_eq,h.prompt]

theorem iterate_eq (p : Layout) (r : Regions) (extent : RegionBounds) (M : Algebra α) (dots : Dot α)
    (scale : ℝ) (a b : FPReference.Inputs α) (h : Agree r extent a b) (hb : IOSourceSafety.Bounds p r b extent)
    (cur : Int) (nonnegative : 0≤cur) (aligned : ∃ n : Nat,cur=(n*p.BN : Nat)) (old : Acc p α) :
    iterate p M dots scale a cur ((blockMask p b : Int)*keyEnd p b) old=
      iterate p M dots scale b cur ((blockMask p b : Int)*keyEnd p b) old := by
  conv_lhs => rw [iterate]
  conv_rhs => rw [iterate]
  by_cases zero : p.BN=0
  · simp only [dif_pos zero]
  · simp only [dif_neg zero]
    by_cases live : cur<(blockMask p b : Int)*keyEnd p b
    · simp only [dif_pos live,advance_eq p r extent M dots scale a b h hb cur nonnegative aligned live]
      apply iterate_eq p r extent M dots scale a b h hb (cur+p.BN) (by omega) ?_
      rcases aligned with ⟨n,hn⟩
      exact ⟨n+1,by simp [hn,Nat.add_mul]⟩
    · simp only [dif_neg live]
termination_by ((blockMask p b : Int)*keyEnd p b-cur).toNat
decreasing_by omega

theorem output_eq (p : Layout) (r : Regions) (extent : RegionBounds) (M : Algebra α) (dots : Dot α)
    (scale : ℝ) (a b : FPReference.Inputs α) (h : Agree r extent a b) (hb : IOSourceSafety.Bounds p r b extent) :
    FPReference.output p M dots scale a=FPReference.output p M dots scale b := by
  simp only [FPReference.output,h.mask_eq,h.end_eq,
    iterate_eq p r extent M dots scale a b h hb 0 (by omega) ⟨0,by simp⟩]

#axiomsClean output_eq
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.IOReferenceAgreement
