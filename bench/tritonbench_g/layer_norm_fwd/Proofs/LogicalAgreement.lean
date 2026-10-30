import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Formula

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.LogicalAgreement
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section

/-- Only the logical values actually used by the selected branches occur. -/
structure Agree (f : Flags) (M N : Nat) (a b : Inputs) : Prop where
  x : ∀ i, i<M → ∀ j, j<N → a.x i j=b.x i j
  x1 : f.x1=Bool.true → ∀ i, i<M → ∀ j, j<N → a.x1 i j=b.x1 i j
  residual : f.residual=Bool.true → ∀ i, i<M → ∀ j, j<N → a.residual i j=b.residual i j
  weight : ∀ j, j<N → a.weight j=b.weight j
  bias : f.bias=Bool.true → ∀ j, j<N → a.bias j=b.bias j
  weight1 : f.w1=Bool.true → ∀ j, j<N → a.weight1 j=b.weight1 j
  bias1 : f.w1=Bool.true → f.b1=Bool.true → ∀ j, j<N → a.bias1 j=b.bias1 j
  rowscale : f.rowscale=Bool.true → ∀ i, i<M → a.rowscale i=b.rowscale i
  rowscale1 : f.x1=Bool.true → f.rowscale=Bool.true → ∀ i, i<M → a.rowscale (M+i)=b.rowscale (M+i)
  seeds : f.dropout=Bool.true → ∀ i, i<M → a.seeds i=b.seeds i
  seeds1 : f.x1=Bool.true → f.dropout=Bool.true → ∀ i, i<M → a.seeds (M+i)=b.seeds (M+i)

namespace Agree
variable {f : Flags} {M N : Nat} {a b : Inputs} (h : Agree f M N a b)
include h

theorem first (p : ℝ) (i j : Nat) (hi : i<M) (hj : j<N) :
    Formula.stream f a M p Bool.false i j=Formula.stream f b M p Bool.false i j := by
  cases hs : f.rowscale <;> cases hd : f.dropout <;>
    simp [Formula.stream,hs,hd,h.x i hi j hj,h.rowscale,h.seeds,hi]

theorem second (p : ℝ) (i j : Nat) (hi : i<M) (hj : j<N) (hx : f.x1=Bool.true) :
    Formula.stream f a M p Bool.true i j=Formula.stream f b M p Bool.true i j := by
  cases hs : f.rowscale <;> cases hd : f.dropout <;>
    simp [Formula.stream,hs,hd,h.x1 hx i hi j hj,h.rowscale1,h.seeds1,hi,hx]

theorem combined (p : ℝ) (i j : Nat) (hi : i<M) (hj : j<N) :
    Formula.combined f a M p i j=Formula.combined f b M p i j := by
  unfold Formula.combined
  rw [h.first p i j hi hj]
  cases hx : f.x1 <;> cases hr : f.residual <;>
    simp [hx,hr,h.second,h.residual,hi,hj]

theorem mean (p : ℝ) (i : Nat) (hi : i<M) :
    Formula.mean f a M N p i=Formula.mean f b M N p i := by
  unfold Formula.mean
  congr 1
  exact Finset.sum_congr rfl (fun j hj => h.combined p i j hi (Finset.mem_range.mp hj))

theorem centered (p : ℝ) (i j : Nat) (hi : i<M) (hj : j<N) :
    Formula.centered f a M N p i j=Formula.centered f b M N p i j := by
  simp only [Formula.centered,h.combined p i j hi hj,h.mean p i hi]

theorem variance (p : ℝ) (i : Nat) (hi : i<M) :
    Formula.variance f a M N p i=Formula.variance f b M N p i := by
  unfold Formula.variance
  congr 1
  exact Finset.sum_congr rfl (fun j hj => congrArg (fun x : ℝ => x^2)
    (h.centered p i j hi (Finset.mem_range.mp hj)))

theorem rstd (eps p : ℝ) (i : Nat) (hi : i<M) :
    Formula.rstd f a M N eps p i=Formula.rstd f b M N eps p i := by
  simp only [Formula.rstd,h.variance p i hi]

theorem normalized (eps p : ℝ) (i j : Nat) (hi : i<M) (hj : j<N) :
    Formula.normalized f a M N eps p i j=Formula.normalized f b M N eps p i j := by
  simp only [Formula.normalized,h.centered p i j hi hj,h.rstd eps p i hi]

theorem y (eps p : ℝ) (i j : Nat) (hi : i<M) (hj : j<N) :
    Formula.y f a M N eps p i j=Formula.y f b M N eps p i j := by
  unfold Formula.y
  rw [h.normalized eps p i j hi hj,h.weight j hj]
  cases hb : f.bias <;> simp [hb,h.bias,hj]

theorem y1 (eps p : ℝ) (i j : Nat) (hi : i<M) (hj : j<N) (hw : f.w1=Bool.true) :
    Formula.y1 f a M N eps p i j=Formula.y1 f b M N eps p i j := by
  unfold Formula.y1
  rw [h.normalized eps p i j hi hj,h.weight1 hw j hj]
  cases hb : f.b1 <;> simp [hb,h.bias1,hw,hj]

end Agree
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.LogicalAgreement
