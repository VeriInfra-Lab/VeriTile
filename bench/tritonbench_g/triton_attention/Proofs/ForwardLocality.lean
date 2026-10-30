import bench.tritonbench_g.triton_attention.Proofs.MemoryAgreement

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.MemoryAgreement
open VeriTile Triton FP.Structural Source
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 3000000
set_option maxRecDepth 32000

theorem forward_advance (P : Primitives α) (p : ForwardParameters)
    (a b : State α) (ids : a.pids=b.pids) (block : Nat) (acc : ForwardAccumulator α p)
    (query : forwardQ p a=forwardQ p b)
    (keys : forwardKV p a p.k block=forwardKV p b p.k block)
    (values : forwardKV p a p.v block=forwardKV p b p.v block) :
    forwardAdvance P p a block acc=forwardAdvance P p b block acc := by
  have dot : forwardDot P p a block=forwardDot P p b block := by
    unfold forwardDot
    rw [query,keys]
  have score : forwardScore P p a block=forwardScore P p b block := by
    funext i
    simp only [forwardScore,forwardQueryRow,ids,dot]
    rfl
  have maximum : forwardMax P p a block acc=forwardMax P p b block acc := by
    funext i
    simp only [forwardMax,score]
  have old : forwardOldDenominator P p a block acc=forwardOldDenominator P p b block acc := by
    funext i
    simp only [forwardOldDenominator,maximum]
  have weights : forwardWeights P p a block acc=forwardWeights P p b block acc := by
    funext i
    simp only [forwardWeights,score,maximum]
  have denominator : forwardDenominator P p a block acc=forwardDenominator P p b block acc := by
    funext i
    simp only [forwardDenominator,weights,old]
  have reciprocal : forwardReciprocal P p a block acc=forwardReciprocal P p b block acc := by
    funext i
    simp only [forwardReciprocal,denominator]
  have normalized : forwardNormalized P p a block acc=forwardNormalized P p b block acc := by
    funext i
    simp only [forwardNormalized,weights,reciprocal]
  have half : forwardHalfWeights P p a block acc=forwardHalfWeights P p b block acc := by
    funext i
    simp only [forwardHalfWeights,normalized]
  have rescaled : forwardRescaled P p a block acc=forwardRescaled P p b block acc := by
    funext i
    simp only [forwardRescaled,old,reciprocal]
  have valueDot : forwardValueDot P p a block acc=forwardValueDot P p b block acc := by
    unfold forwardValueDot
    rw [half,values]
  unfold forwardAdvance
  rw [maximum,denominator,rescaled,valueDot]

theorem forward_rounds (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (legal : Grid.Legal (Grid.forwardGrid c) pid) (a b : Memory α) (h : Agree c.elements a b)
    (n : Nat) (live : n≤forwardBlocks (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid a)) :
    forwardRounds P (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid a) n=
      forwardRounds P (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid b) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [forwardRounds,forwardRounds,ih (by omega)]
    exact forward_advance P (forwardParameters c scale)
      (Grid.entry (Grid.forwardGrid c) pid a) (Grid.entry (Grid.forwardGrid c) pid b) rfl n _
      (forward_query c scale pid legal a b h)
      (forward_kv c scale pid legal a b h "K" n (by omega))
      (forward_kv c scale pid legal a b h "V" n (by omega))

/-- The existing input-derived recurrence reads only the finite source
windows. No source execution theorem is assumed or reconstructed here. -/
theorem forward_step (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (legal : Grid.Legal (Grid.forwardGrid c) pid) (a b : Memory α) (h : Agree c.elements a b) :
    Agree c.elements (Grid.forwardStep P c scale pid a) (Grid.forwardStep P c scale pid b) := by
  have rounds := forward_rounds P c scale pid legal a b h
    (forwardBlocks (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid a)) (Nat.le_refl _)
  have output : forwardOutput P (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid a)=
      forwardOutput P (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid b) := by
    funext i
    exact congrArg (fun acc=>Cell.mk .fp32 (P.cast none .fp16 .fp32
      (P.cast none .real .fp16 (acc.numerator i)))) rounds
  have denominator : forwardLCell P (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid a)=
      forwardLCell P (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid b) := by
    funext i
    exact congrArg (fun acc=>Cell.mk .fp32 (P.cast none .real .fp32 (acc.denominator i))) rounds
  have maximum : forwardMCell P (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid a)=
      forwardMCell P (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid b) := by
    funext i
    exact congrArg (fun acc=>Cell.mk .fp32 (P.cast none .real .fp32 (acc.maximum i))) rounds
  unfold Grid.forwardStep forwardWrite
  dsimp only []
  rw [output,denominator,maximum]
  apply maskedScatterState
  apply scatterState
  exact scatterState _ _ _ h _ _ _ _

#axiomsClean forward_step
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.MemoryAgreement
