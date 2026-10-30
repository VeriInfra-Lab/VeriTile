import bench.tritonbench_g.attention_llama.Proofs.SourceExecution
import bench.tritonbench_g.attention_llama.Proofs.MaskCoverage

namespace VeriTile.Bench.Repairs.AttentionLlama.TileMath
open VeriTile.Triton
open MaskCoverage
open scoped BigOperators
set_option maxHeartbeats 1000000

noncomputable def weight (score : WithBot ℝ) (maximum : ℝ) : ℝ :=
  match score with
  | none => 0
  | some s => Real.exp (s - maximum)

theorem exp_sub_finite (score : WithBot ℝ) (maximum : ℝ) :
    WithBot.realExp (WithBot.realSub score (some maximum)) = some (weight score maximum) := by
  cases score <;> rfl

theorem masked_exp_rowsum {BM BN : Nat}
    (qkT : Tile .real [BM, BN]) (mNewT : Tile .real [BM]) (μ : Fin BM → ℝ)
    (hμ : ∀ r, mNewT.data (r, PUnit.unit) = some (μ r)) (r : Fin BM) :
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length)
        (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
            qkT (Tile.expandDim ⟨1, by simp⟩ mNewT)))).data (r, PUnit.unit)
      = some (∑ j : Fin BN, weight (qkT.data (r, j, PUnit.unit)) (μ r)) := by
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ k : Fin (TileShape.axisDim [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length)),
      (Tile.uop WithBot.realExp
        (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
          qkT (Tile.expandDim ⟨1, by simp⟩ mNewT))).data
        (TileShape.insertAxisIndex [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length) (r, PUnit.unit) k)
      = (some (weight (qkT.data (r, k, PUnit.unit)) (μ r)) : WithBot ℝ) := by
    intro k
    rw [show (TileShape.insertAxisIndex [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length)
          (r, PUnit.unit) k) = (r, k, PUnit.unit) from rfl]
    rw [Tile.uop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.sub]
    rw [hμ r, exp_sub_finite]
  rw [Finset.sum_congr rfl (fun k _ => hcell k), WithBot.sum_someTerm_eq_some]
  rfl

theorem masked_value_dot {BM BN BD : Nat}
    (qkT : Tile .real [BM, BN]) (mNewT lrcpT : Tile .real [BM]) (vT : Tile .real [BN, BD])
    (μ : Fin BM → ℝ) (vfn : TileIndex [BN, BD] → ℝ)
    (hμ : ∀ r, mNewT.data (r, PUnit.unit) = some (μ r))
    (hv : ∀ idx, vT.data idx = some (vfn idx)) (r : Fin BM) (d : Fin BD) (reciprocal : ℝ)
    (hlr : lrcpT.data (r, PUnit.unit) = some reciprocal) :
    (Tile.dot []
        (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
              qkT (Tile.expandDim ⟨1, by simp⟩ mNewT)))
          (Tile.expandDim ⟨1, by simp⟩ lrcpT)) vT).data (r, d, PUnit.unit)
      = some (∑ j : Fin BN,
          (weight (qkT.data (r, j, PUnit.unit)) (μ r) * reciprocal) * vfn (j, d, PUnit.unit)) := by
  rw [Tile.dot_nil_data]
  refine Eq.trans (b := (∑ j : Fin BN, (some
      ((weight (qkT.data (r, j, PUnit.unit)) (μ r) * reciprocal) *
        vfn (j, d, PUnit.unit)) : WithBot ℝ) : WithBot ℝ))
      (Finset.sum_congr rfl (fun j _ => ?_)) ?_
  · rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.mul]
    rw [Tile.uop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.sub]
    rw [hμ r, hlr, hv (j, d, PUnit.unit), exp_sub_finite]
    rfl
  · exact WithBot.sum_someTerm_eq_some _ _

theorem blockKeys_image (N : Nat) (causal : Bool) (position query block : Nat) :
    blockKeys N causal position query block =
      ((Finset.univ : Finset (Fin 64)).filter (fun lane =>
        block * 64 + lane.val < N ∧ visible causal position query (block * 64 + lane.val))).image
        (fun lane => block * 64 + lane.val) := by
  ext key
  simp only [blockKeys, logicalKeys, Finset.mem_filter, Finset.mem_range,
    Finset.mem_image, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨⟨hn, hv⟩, hlo, hhi⟩
    refine ⟨⟨key - block * 64, by omega⟩, ?_, ?_⟩
    swap
    · change block * 64 + (key - block * 64) = key
      omega
    have he : block * 64 + (key - block * 64) = key := by omega
    simpa only [he] using And.intro hn hv
  · rintro ⟨lane, ⟨hn, hv⟩, rfl⟩
    exact ⟨⟨hn, hv⟩, by omega, by omega⟩

theorem sum_masked_lanes (g : Nat → ℝ) (N : Nat) (causal : Bool)
    (position query block : Nat) :
    (∑ lane : Fin 64, if block * 64 + lane.val < N ∧
        visible causal position query (block * 64 + lane.val)
      then g (block * 64 + lane.val) else 0) =
      ∑ key ∈ blockKeys N causal position query block, g key := by
  rw [blockKeys_image, Finset.sum_image]
  · rw [Finset.sum_filter]
  · intro a _ b _ hab
    apply Fin.ext
    change block * 64 + a.val = block * 64 + b.val at hab
    omega

theorem weights_sum (g : Nat → ℝ) (N : Nat) (causal : Bool)
    (position query block : Nat) (m : ℝ) :
    (∑ lane : Fin 64, weight
      (if block * 64 + lane.val < N ∧ visible causal position query (block * 64 + lane.val)
        then some (g (block * 64 + lane.val)) else none) m) =
      ∑ key ∈ blockKeys N causal position query block, Real.exp (g key - m) := by
  convert sum_masked_lanes (fun key => Real.exp (g key - m)) N causal position query block using 1
  apply Finset.sum_congr rfl
  intro lane _
  split <;> rfl

theorem weights_dot (g v : Nat → ℝ) (N : Nat) (causal : Bool)
    (position query block : Nat) (m reciprocal : ℝ) :
    (∑ lane : Fin 64, (weight
      (if block * 64 + lane.val < N ∧ visible causal position query (block * 64 + lane.val)
        then some (g (block * 64 + lane.val)) else none) m * reciprocal) *
      (if block * 64 + lane.val < N then v (block * 64 + lane.val) else 0)) =
      ∑ key ∈ blockKeys N causal position query block,
        (Real.exp (g key - m) * reciprocal) * v key := by
  convert sum_masked_lanes (fun key => (Real.exp (g key - m) * reciprocal) * v key)
    N causal position query block using 1
  apply Finset.sum_congr rfl
  intro lane _
  by_cases h : block * 64 + lane.val < N ∧ visible causal position query (block * 64 + lane.val)
  · simp [h, h.1, weight]
  · simp [h, weight]

theorem previous_scale (old : WithBot ℝ) (oldMaximum oldDenominator m : ℝ)
    (h : old = some oldMaximum ∨ oldDenominator = 0) :
    WithBot.realMul (some oldDenominator)
      (WithBot.realExp (WithBot.realSub old (some m))) =
        some (oldDenominator * Real.exp (oldMaximum - m)) := by
  rcases h with h | rfl
  · rw [h]; rfl
  · cases old using Option.rec <;>
      simp [WithBot.realMul, WithBot.realExp, WithBot.realSub, Option.map₂]

theorem finite_max {BM : Nat} (qk : Tile .real [BM, 64])
    (rm old : Tile .real [BM])
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, 64].length) qk = some rm)
    (hfinite : ∀ r : Fin BM, (∃ x, old.data (r, PUnit.unit) = some x) ∨
      ∃ j : Fin 64, ∃ x, qk.data (r, j, PUnit.unit) = some x) :
    ∃ μ : Fin BM → ℝ, ∀ r,
      (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame rm old) rm old).data
        (r, PUnit.unit) = some (μ r) := by
  have hex : ∀ r : Fin BM, ∃ x,
      (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame rm old) rm old).data
        (r, PUnit.unit) = some x := by
    intro r
    have hs := VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel.al_rmax_cell
      (by decide : 0 < 64) qk rm hrm r
    have hn : rm.data (r, PUnit.unit) ≠ none ∨ old.data (r, PUnit.unit) ≠ none := by
      rcases hfinite r with ⟨x, hx⟩ | ⟨j, x, hx⟩
      · exact Or.inr (by rw [hx]; simp)
      · left
        rw [hs]
        intro he
        have hl := Finset.le_sup (f := fun j : Fin 64 => qk.data (r, j, PUnit.unit))
          (Finset.mem_univ j)
        dsimp only at hl
        rw [hx, he] at hl
        exact (not_le_of_gt (WithBot.bot_lt_coe x)) hl
    simp only [Tile.select_data, Tile.cop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      ComparableDType.gt]
    have hmax : ∀ a b : WithBot ℝ, a ≠ none ∨ b ≠ none →
        ∃ x, (if decide (a > b) = Bool.true then a else b) = some x := by
      intro a b h
      cases a <;> cases b
      · rcases h with h | h <;> exact (h rfl).elim
      · simp only [gt_iff_lt, not_lt_bot, decide_false, Bool.false_eq_true, if_false]
        exact ⟨_, rfl⟩
      · simp only [gt_iff_lt, WithBot.bot_lt_coe, decide_true, if_true]
        exact ⟨_, rfl⟩
      · split <;> exact ⟨_, rfl⟩
    exact hmax _ _ hn
  choose μ hμ using hex
  exact ⟨μ, hμ⟩

end VeriTile.Bench.Repairs.AttentionLlama.TileMath
