import bench.tritonbench_g.flash_attn.Proofs.FlashAttn_Real_RealStep
import bench.tritonbench_g.flash_attn.Proofs.FlashAttn_Grid
-- Preserve the historical Bool spellings after importing the shared Triton DSL.
local notation "true" => Bool.true
local notation "false" => Bool.false
namespace VeriTile.Bench.Repairs.FlashAttn.Real
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000

theorem prefix_nonempty (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row col n : Nat)
    (hn : 0<x.shape.length) (hc : 0<n) : prefixPairs x m scale b h row col n≠[] := by
  have hz : 0∈prefixKeys x.shape row n := by simp [prefixKeys,visible,hn];omega
  have hm:=List.mem_map_of_mem (f:=fun k=>(Spec.score x m scale b h row k,logical m .v x.v b h k col)) hz
  intro he
  change _∈prefixPairs x m scale b h row col n at hm
  simpa [he] using hm

theorem maximum_append (xs ys : List (ℝ × ℝ)) :
    OnlineMath.maximum xs ⊔ OnlineMath.maximum ys=OnlineMath.maximum (xs++ys) := by
  have hh:=congrArg Prod.fst (OnlineMath.running_append_block xs ys)
  simpa only [OnlineMath.running_max] using hh

theorem prefix_invariant (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (b h row col n : Nat) (hr : row<x.shape.length) :
    (List.range n).foldl (blockStep x ops (lift m) (some scale) b h row col) (initial ops)=
      embed (OnlineMath.running (prefixPairs x m scale b h row col n)) := by
  induction n with
  | zero=>rfl
  | succ n ih=>
    simp only [List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil,ih,prefix_succ]
    apply block_eq x hs m scale b h row col n hr
    rw [maximum_append,←prefix_succ]
    exact OnlineMath.maximum_finite _ (prefix_nonempty x m scale b h row col (n+1) hs.2.2.1 (by omega))

theorem filter_range_extend (p : Nat → Bool) (a b : Nat) (hab : a≤b)
    (bounded : ∀ k,p k=true → k<a) : (List.range b).filter p=(List.range a).filter p := by
  rw [show b=a+(b-a) by omega,List.range_add,List.filter_append]
  have tail : (((List.range (b-a)).map (fun j=>a+j)).filter p)=[] := by
    apply List.filter_eq_nil_iff.mpr
    intro k hk
    obtain ⟨j,hj,rfl⟩:=List.mem_map.mp hk
    cases he : p (a+j)
    · simp
    · have:=bounded (a+j) he
      omega
  simp [tail]

theorem prefix_full (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row col : Nat)
    (hr : row<x.shape.length) :
    prefixPairs x m scale b h row col (rounds x.shape (row/128))=Spec.pairs x m scale b h row col := by
  have boundN : ∀ k,visible x.shape row k=true → k<x.shape.length := by
    intro k hk
    have hh : k<x.shape.length ∧ (x.shape.causal=false ∨ k≤row) := by simpa [visible] using hk
    exact hh.1
  have boundM:=visible_visited x.shape row
  have eq : (List.range (rounds x.shape (row/128)*64)).filter (visible x.shape row)=
      (List.range x.shape.length).filter (visible x.shape row) := by
    rcases Nat.le_total x.shape.length (rounds x.shape (row/128)*64) with h|h
    · exact filter_range_extend _ _ _ h boundN
    · exact (filter_range_extend _ _ _ h (fun k hk=>visible_visited x.shape row k hr hk)).symm
  simp only [prefixPairs,prefixKeys,Spec.pairs,Spec.keys,eq]

theorem accumulated_real (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (b h row col : Nat) (hr : row<x.shape.length) :
    accumulated x ops (lift m) (some scale) b h row col=
      embed (OnlineMath.running (Spec.pairs x m scale b h row col)) := by
  rw [accumulated,prefix_invariant x hs m scale b h row col _ hr,prefix_full x m scale b h row col hr]

theorem output_real (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (b h row col : Nat) (hr : row<x.shape.length) :
    outputValue x ops (lift m) (some scale) b h row col=some (Spec.output x m scale b h row col) := by
  rw [outputValue,accumulated_real x hs m scale b h row col hr]
  change some ((OnlineMath.running (Spec.pairs x m scale b h row col)).2.2/
    (OnlineMath.running (Spec.pairs x m scale b h row col)).2.1)=_
  rw [OnlineMath.ratio_eq_reference _ (Spec.pairs_nonempty x m scale b h row col hs.2.2.1)]
  simp only [Spec.output,Spec.numerator,Spec.denominator,OnlineMath.numerator,OnlineMath.denominator,
    Spec.pairs,List.map_map,Function.comp_def]

theorem log_real (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (b h row : Nat) (hr : row<x.shape.length) :
    logValue x ops (lift m) (some scale) b h row=some (Spec.lse x m scale b h row) := by
  let xs:=Spec.pairs x m scale b h row 0
  have hn:=Spec.pairs_nonempty x m scale b h row 0 hs.2.2.1
  obtain ⟨z,hz⟩:=OnlineMath.maximum_finite xs hn
  rw [logValue,accumulated_real x hs m scale b h row 0 hr]
  change WithBot.realAdd (OnlineMath.running xs).1 (WithBot.realLog2 (some (OnlineMath.running xs).2.1))=_
  rw [OnlineMath.running_max,hz]
  change some (z+Real.log (OnlineMath.running xs).2.1/Real.log 2)=_
  have he:=OnlineMath.logsumexp_eq_reference xs hn
  rw [hz] at he
  change z+Real.log (OnlineMath.running xs).2.1/Real.log 2=_ at he
  rw [he]
  simp only [Spec.lse,Spec.denominator,OnlineMath.denominator,xs,Spec.pairs,List.map_map,Function.comp_def]

theorem whole_output (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x) (m : Memory ℝ)
    (scale : ℝ) (ws : List Work) (complete : Complete x.shape ws) (b h row col : Nat)
    (active : (Work.output b h row col).active x.shape) :
    execute x ops (some scale) ws (lift m) .o (address x.o b h row col)=
      some (Spec.output x m scale b h row col) := by
  have hh:=call_value x hs ho ops (some scale) ws complete (lift m) (.output b h row col) active
  exact hh.trans (output_real x hs m scale b h row col active.2.2.1)

theorem whole_logsum (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x) (m : Memory ℝ)
    (scale : ℝ) (ws : List Work) (complete : Complete x.shape ws) (b h row : Nat)
    (active : (Work.logsum b h row).active x.shape) :
    execute x ops (some scale) ws (lift m) .l (lAddress x.shape b h row)=
      some (Spec.lse x m scale b h row) := by
  have hh:=call_value x hs ho ops (some scale) ws complete (lift m) (.logsum b h row) active
  exact hh.trans (log_real x hs m scale b h row active.2.2)
#axiomsClean prefix_invariant
#axiomsClean prefix_full
#axiomsClean output_real
#axiomsClean log_real
#axiomsClean whole_output
#axiomsClean whole_logsum
end
end VeriTile.Bench.Repairs.FlashAttn.Real
