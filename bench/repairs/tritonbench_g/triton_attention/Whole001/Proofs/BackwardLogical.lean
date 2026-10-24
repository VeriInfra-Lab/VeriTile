import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.CausalBlocks

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001
open scoped BigOperators
namespace Math

/-- Exactly the row constructed by a 128-row tile and its lane. -/
def tileRow (c : Shape) (b : Fin c.blocks) (r : Fin 128) : Fin c.rows :=
  finProdFinEquiv (b,r)

theorem tileRow_val (c : Shape) (b : Fin c.blocks) (r : Fin 128) :
    (tileRow c b r).val = b.val * 128 + r.val := by
  simp [tileRow,finProdFinEquiv,Nat.mul_comm,Nat.add_comm]

theorem tileRow_block (c : Shape) (b : Fin c.blocks) (r : Fin 128) :
    (tileRow c b r).val / 128 = b.val := by
  rw [tileRow_val]
  omega

theorem tiled_sum (c : Shape) (f : Fin c.rows → ℝ) :
    (∑ b : Fin c.blocks, ∑ r : Fin 128, f (tileRow c b r)) = ∑ j, f j := by
  calc
    _ = ∑ p : Fin c.blocks × Fin 128, f (finProdFinEquiv p) :=
      (Fintype.sum_prod_type (fun p : Fin c.blocks × Fin 128 => f (finProdFinEquiv p))).symm
    _ = _ := Fintype.sum_equiv finProdFinEquiv _ _ (fun _ => rfl)

noncomputable def lowerTiledSum (c : Shape) (i : Fin c.rows) (f : Fin c.rows → ℝ) : ℝ :=
  ∑ b : Fin c.blocks, ∑ r : Fin 128,
    if b.val ≤ i.val / 128 then f (tileRow c b r) else 0

noncomputable def upperTiledSum (c : Shape) (j : Fin c.rows) (f : Fin c.rows → ℝ) : ℝ :=
  ∑ b : Fin c.blocks, ∑ r : Fin 128,
    if j.val / 128 ≤ b.val then f (tileRow c b r) else 0

theorem lowerTiledSum_eq_sum (c : Shape) (i : Fin c.rows) (f : Fin c.rows → ℝ)
    (hz : ∀ j, ¬j ≤ i → f j = 0) : lowerTiledSum c i f = ∑ j, f j := by
  rw [← tiled_sum c f]
  apply Finset.sum_congr rfl
  intro b _
  apply Finset.sum_congr rfl
  intro r _
  dsimp only
  split_ifs with h
  · rfl
  · symm
    apply hz
    intro hji
    have : (tileRow c b r).val ≤ i.val := hji
    have hb := tileRow_block c b r
    omega

theorem upperTiledSum_eq_sum (c : Shape) (j : Fin c.rows) (f : Fin c.rows → ℝ)
    (hz : ∀ i, ¬j ≤ i → f i = 0) : upperTiledSum c j f = ∑ i, f i := by
  rw [← tiled_sum c f]
  apply Finset.sum_congr rfl
  intro b _
  apply Finset.sum_congr rfl
  intro r _
  dsimp only
  split_ifs with h
  · rfl
  · symm
    apply hz
    intro hji
    have : j.val ≤ (tileRow c b r).val := hji
    have hb := tileRow_block c b r
    omega

/-- The actual source backward block traversal after erasing numerical
operations: DQ sees increasing key blocks; DK/DV see query blocks from the
current key block through the end. The scalar causal mask remains active
inside the diagonal block. All saved tensors come from the forward loop. -/
noncomputable def tiledDQ (c : Shape) (Q K V G : Tensor c.rows c.dim) (scale : ℝ) :
    Tensor c.rows c.dim := fun i e => lowerTiledSum c i (fun j =>
  sourceDS Q K V G (sourceForwardOutput Q K V scale) scale
    (sourceForwardL Q K V scale) (sourceForwardM Q K V scale) i j * K j e)
noncomputable def tiledDK (c : Shape) (Q K V G : Tensor c.rows c.dim) (scale : ℝ) :
    Tensor c.rows c.dim := fun j e => upperTiledSum c j (fun i =>
  sourceDS Q K V G (sourceForwardOutput Q K V scale) scale
    (sourceForwardL Q K V scale) (sourceForwardM Q K V scale) i j * Q i e)
noncomputable def tiledDV (c : Shape) (Q K V G : Tensor c.rows c.dim) (scale : ℝ) :
    Tensor c.rows c.dim := fun j e => upperTiledSum c j (fun i =>
  sourceP Q K scale (sourceForwardM Q K V scale) i j *
    preprocessDO G (sourceForwardL Q K V scale) i e)

theorem tiledDQ_eq_sourceDQ (c : Shape) (Q K V G : Tensor c.rows c.dim) (scale : ℝ) :
    tiledDQ c Q K V G scale = sourceDQ Q K V G (savedOutput Q K V scale) scale
      (savedNormalizer Q K scale) (savedMax Q K scale) := by
  obtain ⟨hy,hl,hm⟩ := sourceForward_tensors Q K V scale
  funext i e
  simp only [tiledDQ,hy,hl,hm,sourceDQ]
  apply lowerTiledSum_eq_sum
  intro j hj
  simp [sourceDS,sourceP,hj]

theorem tiledDK_eq_sourceDK (c : Shape) (Q K V G : Tensor c.rows c.dim) (scale : ℝ) :
    tiledDK c Q K V G scale = sourceDK Q K V G (savedOutput Q K V scale) scale
      (savedNormalizer Q K scale) (savedMax Q K scale) := by
  obtain ⟨hy,hl,hm⟩ := sourceForward_tensors Q K V scale
  funext j e
  simp only [tiledDK,hy,hl,hm,sourceDK]
  apply upperTiledSum_eq_sum
  intro i hi
  simp [sourceDS,sourceP,hi]

theorem tiledDV_eq_sourceDV (c : Shape) (Q K V G : Tensor c.rows c.dim) (scale : ℝ) :
    tiledDV c Q K V G scale = sourceDV Q K G scale
      (savedNormalizer Q K scale) (savedMax Q K scale) := by
  obtain ⟨_,hl,hm⟩ := sourceForward_tensors Q K V scale
  funext j e
  simp only [tiledDV,hl,hm,sourceDV]
  apply upperTiledSum_eq_sum
  intro i hi
  simp [sourceP,hi]

/-- Full logical forward and backward tensor connection for every supported
shape and every real input. This is a mathematical interpretation theorem,
with no whole-output, dot, reduction, or saved-value premise. -/
theorem whole_logical_tensors (c : Shape) (Q K V G : Tensor c.rows c.dim) (scale : ℝ) :
    sourceForwardOutput Q K V scale = attention Q K V scale ∧
    tiledDQ c Q K V G scale = gradientQ Q K V G scale ∧
    tiledDK c Q K V G scale = gradientK Q K V G scale ∧
    tiledDV c Q K V G scale = gradientV Q K G scale := by
  rw [(sourceForward_tensors Q K V scale).1,tiledDQ_eq_sourceDQ,
    tiledDK_eq_sourceDK,tiledDV_eq_sourceDV]
  exact source_backward_eq_gradients Q K V G scale

#axiomsClean tiled_sum
#axiomsClean lowerTiledSum_eq_sum
#axiomsClean upperTiledSum_eq_sum
#axiomsClean whole_logical_tensors
#specNonCircular gradientQ avoiding [tiledDQ,lowerTiledSum]
#specNonCircular gradientK avoiding [tiledDK,upperTiledSum]
#specNonCircular gradientV avoiding [tiledDV,upperTiledSum]
end Math
end VeriTile.Bench.Repairs.TritonAttention.Whole001
