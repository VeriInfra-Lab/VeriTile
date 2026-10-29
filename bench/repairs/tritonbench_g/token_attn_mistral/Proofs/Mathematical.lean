import bench.repairs.tritonbench_g.token_attn_mistral.Host
import bench.repairs.tritonbench_g.token_attn_mistral.Real
namespace VeriTile.Bench.Repairs.TokenAttnMistral.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section
def target (c : Call) (m : Memory ℝ) (b h d : Nat) : ℝ :=
  ∑ n ∈ Finset.range (seq c m b),
    if n+windowAt c m b<fullLength c m b then
    (m c.probability.region (h*c.probability.rowStride+(startAt c m b+n)*c.probability.colStride)).read .fp32 *
    (m c.value.region (pageAt c m b n*c.value.slotStride+(h/group c)*c.value.headStride+d*c.value.dimStride)).read .fp32
    else 0

theorem weighted_target (c : Call) (m : Memory ℝ) (b h d : Nat) :
    Real.weighted (params c) (state m (b,h)) d=target c m b h d := by
  simp [Real.weighted,params,state,count,startIndex,requestIndex,probOffset,gatherOffset,valueOffset,
    target,seq,startAt,requestAt,pageAt,sequenceLength,windowStart,fullLength,windowAt,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem whole_real (c : Call) (m : Memory ℝ) (layout : Layout c) :
    ∃ result, invoke Real.algebra Real.sum c m=some result ∧ result.returned=none ∧
      result.output=c.output ∧
      (∀ (b : Fin (batches c)) (h : Fin (heads c)) (d : Fin c.value.width),
        (result.memory c.output.region ((b.val*heads c+h.val)*c.value.width+d.val)).read .fp32=
          target c m b.val h.val d.val) ∧ Frame c m result.memory := by
  obtain ⟨result,run,ret,out,_,readback,frame⟩ := whole_typed Real.algebra Real.sum c m layout
  refine ⟨result,run,ret,out,?_,frame⟩
  intro b h d
  let e : Event (params c) := ((b.val,h.val),(d,PUnit.unit))
  have live : e∈outputEvents c := (events_exact c e).mpr ⟨b.isLt,h.isLt⟩
  have cell := readback e live
  rw [outputOffset_dense c layout] at cell
  rw [cell,Cell.read_same]
  change outputValues Real.algebra Real.sum (params c) (state m (b.val,h.val)) (d,PUnit.unit)=_
  rw [Real.output_weighted _ _ (by change 0<128; decide),weighted_target]


#axiomsClean whole_real
#axiomsClean weighted_target
end
end VeriTile.Bench.Repairs.TokenAttnMistral.Host
