import bench.tritonbench_g.attention_llama.Proofs.TileMath

namespace VeriTile.Bench.Repairs.AttentionLlama.TileStep
open VeriTile.Triton MaskCoverage TileMath
open scoped BigOperators
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

noncomputable def prior {BM : Nat} (oldM oldL newM : Tile .real [BM]) : Tile .real [BM] :=
  Tile.bop NumericDType.real.mul Broadcast.nil.consSame oldL
    (Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame oldM newM))

noncomputable def probability {BM : Nat} (qk : Tile .real [BM,64])
    (newM : Tile .real [BM]) : Tile .real [BM,64] :=
  Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
    qk (Tile.expandDim ⟨1, by simp⟩ newM))

noncomputable def denominator {BM : Nat} (qk : Tile .real [BM,64])
    (oldM oldL newM : Tile .real [BM]) : Tile .real [BM] :=
  Tile.bop NumericDType.real.add Broadcast.nil.consSame
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,64].length) (probability qk newM))
    (prior oldM oldL newM)

noncomputable def reciprocal {BM : Nat} (l : Tile .real [BM]) : Tile .real [BM] :=
  Tile.bop NumericDType.real.div Broadcast.scalarL (Tile.scalar (some (1 : ℝ))) l

noncomputable def normalized {BM : Nat} (qk : Tile .real [BM,64])
    (newM lrcp : Tile .real [BM]) : Tile .real [BM,64] :=
  Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame (probability qk newM)
    (Tile.expandDim ⟨1, by simp⟩ lrcp)

noncomputable def rescaled {BM D : Nat} (acc : Tile .real [BM,D])
    (prev lrcp : Tile .real [BM]) : Tile .real [BM,D] :=
  Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acc
    (Tile.expandDim ⟨1, by simp⟩ (Tile.bop NumericDType.real.mul Broadcast.nil.consSame prev lrcp))

noncomputable def accumulator {BM D : Nat} (qk : Tile .real [BM,64])
    (oldM oldL newM : Tile .real [BM]) (acc : Tile .real [BM,D])
    (v : Tile .real [64,D]) : Tile .real [BM,D] :=
  let lr := reciprocal (denominator qk oldM oldL newM)
  Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
    (rescaled acc (prior oldM oldL newM) lr) (Tile.dot [] (normalized qk newM lr) v)

theorem prior_cell {BM : Nat} (oldM oldL newM : Tile .real [BM])
    (r : Fin BM) (m l μ : ℝ)
    (hm : oldM.data (r, PUnit.unit) = some m ∨ l = 0)
    (hl : oldL.data (r, PUnit.unit) = some l)
    (hμ : newM.data (r, PUnit.unit) = some μ) :
    (prior oldM oldL newM).data (r, PUnit.unit) = some (l * Real.exp (m - μ)) := by
  simp only [prior, Tile.bop_data, Tile.uop_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, NumericDType.sub]
  rw [hl, hμ]
  exact previous_scale _ _ _ _ hm

theorem denominator_cell {BM : Nat} (qk : Tile .real [BM,64])
    (oldM oldL newM : Tile .real [BM]) (μ : Fin BM → ℝ)
    (hμ : ∀ r, newM.data (r, PUnit.unit) = some (μ r))
    (N : Nat) (causal : Bool) (position query block : Nat)
    (g : Nat → ℝ) (r : Fin BM) (m l a : ℝ)
    (hg : ∀ j : Fin 64, qk.data (r,j,PUnit.unit) =
      if block * 64 + j.val < N ∧ visible causal position query (block * 64 + j.val)
        then some (g (block * 64 + j.val)) else none)
    (hm : oldM.data (r, PUnit.unit) = some m ∨ l = 0)
    (hl : oldL.data (r, PUnit.unit) = some l) :
    (denominator qk oldM oldL newM).data (r, PUnit.unit) =
      some ((TailMath.step g (fun _ => 0) (blockKeys N causal position query block)
        (μ r) ⟨m,l,a⟩).denominator) := by
  simp only [denominator, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add]
  rw [prior_cell oldM oldL newM r m l (μ r) hm hl (hμ r)]
  unfold probability
  erw [masked_exp_rowsum qk newM μ hμ r]
  simp only [hg]
  erw [weights_sum g N causal position query block (μ r)]
  rfl

theorem accumulator_cell {BM D : Nat} (qk : Tile .real [BM,64])
    (oldM oldL newM : Tile .real [BM]) (acc : Tile .real [BM,D])
    (v : Tile .real [64,D]) (μ : Fin BM → ℝ)
    (hμ : ∀ r, newM.data (r, PUnit.unit) = some (μ r))
    (N : Nat) (causal : Bool) (position query block : Nat)
    (g value : Nat → ℝ) (r : Fin BM) (d : Fin D) (m l a : ℝ)
    (hg : ∀ j : Fin 64, qk.data (r,j,PUnit.unit) =
      if block * 64 + j.val < N ∧ visible causal position query (block * 64 + j.val)
        then some (g (block * 64 + j.val)) else none)
    (hm : oldM.data (r, PUnit.unit) = some m ∨ l = 0)
    (hl : oldL.data (r, PUnit.unit) = some l)
    (ha : acc.data (r,d,PUnit.unit) = some a)
    (vfn : TileIndex [64,D] → ℝ) (hv : ∀ idx, v.data idx = some (vfn idx))
    (hvd : ∀ j : Fin 64, vfn (j,d,PUnit.unit) =
      if block * 64 + j.val < N then value (block * 64 + j.val) else 0) :
    (accumulator qk oldM oldL newM acc v).data (r,d,PUnit.unit) =
      some ((TailMath.step g value (blockKeys N causal position query block)
        (μ r) ⟨m,l,a⟩).accumulator) := by
  let next := TailMath.step g value (blockKeys N causal position query block) (μ r) ⟨m,l,a⟩
  have hlc : (denominator qk oldM oldL newM).data (r,PUnit.unit) = some next.denominator :=
    denominator_cell qk oldM oldL newM μ hμ N causal position query block g r m l a hg hm hl
  have hlr : (reciprocal (denominator qk oldM oldL newM)).data (r,PUnit.unit) =
      some (1 / next.denominator) := by
    simp only [reciprocal, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar_data, NumericDType.div, hlc]
    rfl
  have hpv := masked_value_dot qk newM (reciprocal (denominator qk oldM oldL newM))
    v μ vfn hμ hv r d (1 / next.denominator) hlr
  simp only [hg, hvd] at hpv
  erw [weights_dot g value N causal position query block (μ r) (1 / next.denominator)] at hpv
  change WithBot.realAdd
    ((rescaled acc (prior oldM oldL newM)
      (reciprocal (denominator qk oldM oldL newM))).data (r,d,PUnit.unit))
    ((Tile.dot [] (normalized qk newM (reciprocal (denominator qk oldM oldL newM))) v).data
      (r,d,PUnit.unit)) = _
  unfold normalized probability
  erw [hpv]
  simp only [rescaled, Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]
  rw [ha, prior_cell oldM oldL newM r m l (μ r) hm hl (hμ r), hlr]
  rfl

end VeriTile.Bench.Repairs.AttentionLlama.TileStep
