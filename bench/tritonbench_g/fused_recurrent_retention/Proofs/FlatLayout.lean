import bench.tritonbench_g.fused_recurrent_retention.Proofs.Events
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section

theorem pair_div (n a b : Nat) (hb : b<n) : (a*n+b)/n=a := by
  have hn : 0<n := by omega
  rw [Nat.add_comm,Nat.add_mul_div_right _ _ hn,Nat.div_eq_of_lt hb,zero_add]

theorem pair_mod (n a b : Nat) (hb : b<n) : (a*n+b)%n=b := by
  simp [Nat.add_mod,Nat.mod_eq_of_lt hb]

theorem pair_bound (outer width a b : Nat) (ha : a<outer) (hb : b<width) :
    a*width+b<outer*width := by
  have he := Nat.mul_le_mul_right width (Nat.succ_le_of_lt ha)
  nlinarith

theorem partialOffset_bound (s : Shape) (count width split head time channel : Nat)
    (hs : split<count) (hh : head<s.BH) (ht : time<s.T) (hc : channel<width) :
    partialOffset s width split head time channel<count*rowCount s width := by
  have h1 := pair_bound count s.BH split head hs hh
  have h2 := pair_bound (count*s.BH) s.T _ time h1 ht
  have h3 := pair_bound (count*s.BH*s.T) width _ channel h2 hc
  simpa only [partialOffset,rowCount,Nat.mul_assoc] using h3

theorem rowAddress_bound (s : Shape) (width head time channel : Nat)
    (hh : head<s.BH) (ht : time<s.T) (hc : channel<width) :
    rowAddress s width head time channel<rowCount s width := by
  exact pair_bound (s.BH*s.T) width _ channel (pair_bound s.BH s.T head time hh ht) hc

theorem flat_partial (s : Shape) (width split head time channel : Nat)
    (hh : head<s.BH) (ht : time<s.T) (hc : channel<width) :
    flatSplit s width (partialOffset s width split head time channel)=split ∧
    flatHead s width (partialOffset s width split head time channel)=head ∧
    flatTime s width (partialOffset s width split head time channel)=time ∧
    flatChannel width (partialOffset s width split head time channel)=channel := by
  simp only [flatSplit,flatHead,flatTime,flatChannel,partialOffset,
    pair_div width _ channel hc,pair_div s.T _ time ht,pair_div s.BH split head hh,
    pair_mod width _ channel hc,pair_mod s.T _ time ht,pair_mod s.BH split head hh]
  simp

theorem row_to_partial (s : Shape) (width split head time channel : Nat) :
    split*rowCount s width+rowAddress s width head time channel=
      partialOffset s width split head time channel := by
  unfold rowCount rowAddress partialOffset
  ring

theorem stateOffset_bound (s : Shape) (head key value : Nat)
    (hh : head<s.BH) (hk : key<s.DK) (hv : value<s.DV) :
    stateOffset s head key value<s.BH*s.DK*s.DV := by
  exact pair_bound (s.BH*s.DK) s.DV _ value (pair_bound s.BH s.DK head key hh hk) hv

theorem SumPlan.eval_congr_on (a : Arithmetic α) (p : SumPlan) (x y : Nat → α)
    (he : ∀ i,i∈p.leaves → x i=y i) : p.eval a x=p.eval a y := by
  induction p with
  | zero => rfl
  | input i => exact he i (by simp [SumPlan.leaves])
  | add l r hl hr =>
    simp only [SumPlan.eval]
    rw [hl (fun i hi => he i (by simp [SumPlan.leaves,hi])),
      hr (fun i hi => he i (by simp [SumPlan.leaves,hi]))]

theorem SumPlan.eval_congr_bound (a : Arithmetic α) (p : SumPlan) (n : Nat) (hp : p.Covers n)
    (x y : Nat → α) (he : ∀ i,i<n → x i=y i) : p.eval a x=p.eval a y := by
  apply p.eval_congr_on a x y
  intro i hi
  exact he i (List.mem_range.mp ((hp.mem_iff).mp hi))

#axiomsClean partialOffset_bound
#axiomsClean flat_partial
#axiomsClean row_to_partial
#axiomsClean stateOffset_bound
#axiomsClean SumPlan.eval_congr_bound
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
