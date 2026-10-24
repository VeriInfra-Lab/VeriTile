import bench.repairs.tritonbench_g.token_attn_reduceV.Stores
namespace VeriTile.Bench.Repairs.TokenAttnReduceV.Grid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000

theorem unique_perm (p : Parameters) (a b : List PID) (perm : a.Perm b) (unique : Unique p a) :
    Unique p b := by
  intro e he f hf eq
  apply unique e _ f _ eq
  · exact (mem_events p a e).mpr (perm.mem_iff.mpr ((mem_events p b e).mp he))
  · exact (mem_events p a f).mpr (perm.mem_iff.mpr ((mem_events p b f).mp hf))

/-- Disjoint stores and immutable input reads make every legal grid scheduling
order observationally identical. No numerical commutativity is involved. -/
theorem schedules_equal {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (a b : List PID) (m : Memory α) (perm : a.Perm b)
    (separate : Separate p) (unique : Unique p a) : graph M R p a m=graph M R p b m := by
  obtain ⟨x,hx,_,xr,xf⟩ := whole M R p a m separate unique
  obtain ⟨y,hy,_,yr,yf⟩ := whole M R p b m separate (unique_perm p a b perm unique)
  have ex : graph M R p a m=x := Option.some.inj ((complete M R p a m).symm.trans hx)
  have ey : graph M R p b m=y := Option.some.inj ((complete M R p b m).symm.trans hy)
  rw [ex,ey]
  funext r o
  by_cases hit : ∃ e∈events p a, r=p.output ∧ o=offset p e.1 e.2
  · obtain ⟨e,he,hr,ho⟩ := hit
    have hb : e∈events p b := (mem_events p b e).mpr (perm.mem_iff.mp ((mem_events p a e).mp he))
    rw [hr,ho,xr e he,yr e hb]
  · have noA : r≠p.output ∨ ∀ e∈events p a,o≠offset p e.1 e.2 := by
      by_cases hr : r=p.output
      · right; intro e he ho; exact hit ⟨e,he,hr,ho⟩
      · exact Or.inl hr
    have noB : r≠p.output ∨ ∀ e∈events p b,o≠offset p e.1 e.2 := by
      rcases noA with hr|ho
      · exact Or.inl hr
      · right; intro e he; exact ho e ((mem_events p a e).mpr (perm.mem_iff.mpr ((mem_events p b e).mp he)))
    rw [xf r o noA,yf r o noB]

#axiomsClean unique_perm
#axiomsClean schedules_equal
end
end VeriTile.Bench.Repairs.TokenAttnReduceV.Grid
