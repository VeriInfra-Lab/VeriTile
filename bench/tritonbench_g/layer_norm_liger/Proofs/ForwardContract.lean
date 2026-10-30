import bench.tritonbench_g.layer_norm_liger.Proofs.ForwardRun
namespace VeriTile.Bench.Repairs.LayerNormLiger.Forward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section

theorem reference_y {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (reference M R p s).mem p.y (outputOffset p s i) = stored M (yValue M R p s i) := by
  apply State.masked_scatter_readback
  · exact hi
  · intro a b hab
    have h : a.1.val = b.1.val := Nat.add_left_cancel hab
    rcases a with ⟨a,⟨⟩⟩
    rcases b with ⟨b,⟨⟩⟩
    simpa [Fin.ext_iff] using h

theorem reference_mean {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (hmy : p.mean ≠ p.y) (hmr : p.mean ≠ p.rstd) :
    (reference M R p s).mem p.mean (meanOffset p s) = stored M (meanValue M R p s) := by
  unfold reference
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hmy)]
  simp [State.write, hmr]

theorem reference_rstd {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (hry : p.rstd ≠ p.y) :
    (reference M R p s).mem p.rstd (rstdOffset p s) = stored M (rstdValue M R p s) := by
  unfold reference
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hry)]
  simp [State.write]

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat) (h : Outside p s r o) :
    (reference M R p s).mem r o = s.mem r o := by
  unfold reference
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ ?_]
  · simp only [State.write]
    rcases h with ⟨hm,hr,_⟩
    have hm' : ¬(r=p.mean ∧ o=meanOffset p s) := by tauto
    have hr' : ¬(r=p.rstd ∧ o=rstdOffset p s) := by tauto
    simp [hm',hr']
  · rcases h.2.2 with hy | hy
    · exact Or.inl hy
    · exact Or.inr (fun i _ hi => hy i hi)

#axiomsClean reference_y
#axiomsClean reference_mean
#axiomsClean reference_rstd
#axiomsClean reference_frame
end
end VeriTile.Bench.Repairs.LayerNormLiger.Forward
