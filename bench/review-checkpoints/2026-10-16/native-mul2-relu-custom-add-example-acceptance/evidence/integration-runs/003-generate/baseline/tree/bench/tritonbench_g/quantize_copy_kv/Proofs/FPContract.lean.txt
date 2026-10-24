import bench.tritonbench_g.quantize_copy_kv.Proofs.FPTypedSource

namespace VeriTile.Bench.TritonBenchG.QuantizeCopyKv.TypedQuantizer
open VeriTile.Triton FP.Structural FP.IntegerConversion
open QuantizeKvTransform
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

private theorem narrow_zero (cfg : FP.Config) :
    MixedBits.narrow cfg (FP.Value.zero .fp32) = FP.Value.zero .fp32 := by
  rcases cfg with ⟨r, fi, fr⟩
  cases r <;> cases fi <;> cases fr <;> decide +kernel

attribute [local irreducible] MixedBits.narrow MixedBits.widen MixedBits.absHalf
  FP.div FPProfile.maximum toSigned totalSigned

/-- Full execution agrees with an independent initial-memory reference for
every input payload and undefined-conversion policy, without a padding or
finite-input premise. Every used numerical callback has a concrete profile. -/
theorem execution_memory (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (s : State Word) (hw : 0 < p.width) :
    (execution rest cfg u p s hw).val.mem = (written cfg u p s).mem := by
  simp +contextual [execution, written, valuesWritten, masked_scatter_memory,
    active, inputOffset, outputOffset, scaleOffset, destination, quantized,
    loaded, scale, maximum, quotient, magnitude, narrow_zero,
    FPProfile.reduceMaximum,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    TileShape.insertAxisIndex, TileShape.replaceAxisIndex, TileShape.dropInsertedIndex, TileShape.eraseAxis,
    show (0.0 : ℝ) = 0 by norm_num, show (127.0 : ℝ) = 127 by norm_num]
  apply congrArg₂ (fun f mem => List.foldl f mem (TileShape.allIndices [p.height, 1]))
  · funext mem i
    by_cases hi : i.1.val < p.heads
    · simp only [hi, ↓reduceIte]
      funext r o
      split_ifs <;> try rfl
      all_goals exfalso; apply_assumption; exact hi
    · simp only [hi, ↓reduceIte]
  · apply congrArg (fun f => List.foldl f s.mem (TileShape.allIndices [p.height, p.width]))
    funext mem i
    by_cases hi : i.1.val < p.heads
    · simp only [hi, ↓reduceIte]
      funext r o
      split_ifs <;> try rfl
      all_goals exfalso; apply_assumption; exact hi
    · simp only [hi, ↓reduceIte]

theorem quantized_defined (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width])
    (h : (toSigned 8 (quotient cfg p s i)).isSome) :
    ∃ q, (quotient cfg p s i).toRat? = some q ∧
      TowardZero q (quantized cfg u p s i) ∧ SignedRange 8 (quantized cfg u p s i) := by
  cases hx : toSigned 8 (quotient cfg p s i) with
  | none => simp [hx] at h
  | some z =>
    have hz := (toSigned_spec 8 (quotient cfg p s i) z).mp hx
    simpa [quantized, totalSigned, hx] using hz

theorem value_readback (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word)
    (i : TileIndex [p.height, p.width]) (hi : active p i)
    (hregions : p.output ≠ p.scales)
    (hinj : Function.Injective (outputOffset p s)) :
    (written cfg u p s).mem p.output (outputOffset p s i) = .mk .int (quantized cfg u p s i) := by
  unfold written
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hregions)]
  exact masked_readback_active _ _ _ _ _ i hi (fun _ _ _ _ he => hinj he)

theorem scale_readback (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word)
    (i : TileIndex [p.height, 1]) (hi : i.1.val < p.heads)
    (hinj : Function.Injective (scaleOffset p s)) :
    (written cfg u p s).mem p.scales (scaleOffset p s i) = .mk .fp16 (scale cfg p s i.1) := by
  exact masked_readback_active _ _ _ _ _ i hi (fun _ _ _ _ he => hinj he)

theorem frame (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word)
    (r : RegionName) (o : Nat)
    (hv : r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width], active p i → o ≠ outputOffset p s i)
    (hs : r ≠ p.scales ∨ ∀ i : TileIndex [p.height, 1], i.1.val < p.heads → o ≠ scaleOffset p s i) :
    (written cfg u p s).mem r o = s.mem r o := by
  unfold written valuesWritten
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (hs.imp id (fun h i _ => h i))]
  exact State.masked_scatter_frame _ _ _ _ _ _ _ (hv.imp id (fun h i _ => h i)) _

/-- Only policy results at active undefined quotients can affect memory.
Inactive-head NaN conversion may choose any value without a padding premise. -/
theorem written_policy_congr (cfg : FP.Config) (u v : Nat → Word → Int)
    (p : Parameters) (s : State Word)
    (h : ∀ i, active p i → toSigned 8 (quotient cfg p s i) = none →
      u 8 (quotient cfg p s i) = v 8 (quotient cfg p s i)) :
    (written cfg u p s).mem = (written cfg v p s).mem := by
  have hv : valuesWritten cfg u p s = valuesWritten cfg v p s := by
    unfold valuesWritten
    congr 1
    funext t i
    by_cases hi : active p i
    · simp only [hi, ↓reduceIte]
      congr 2
      unfold quantized
      cases hx : toSigned 8 (quotient cfg p s i) with
      | none => simp [totalSigned, hx, h i hi hx]
      | some z => simp [totalSigned, hx]
    · simp only [hi, ↓reduceIte]
  simp only [written, hv]

theorem policy_independent (rest : Algebra Word) (cfg : FP.Config) (u v : Nat → Word → Int)
    (p : Parameters) (s : State Word) (hw : 0 < p.width) (h : InputDefined cfg p s) :
    (execution rest cfg u p s hw).val.mem = (execution rest cfg v p s hw).val.mem := by
  rw [execution_memory, execution_memory]
  apply written_policy_congr
  intro i hi hnone
  have hd := h i hi
  simp [hnone] at hd

/-- The residual backend contributes no observed operation to this stage. -/
theorem residual_backend_independent (rest other : Algebra Word) (cfg : FP.Config)
    (u : Nat → Word → Int) (p : Parameters) (s : State Word) (hw : 0 < p.width) :
    (execution rest cfg u p s hw).val.mem = (execution other cfg u p s hw).val.mem := by
  rw [execution_memory, execution_memory]

#specNonCircular written avoiding [source, FP.Structural.exec, execution]
#axiomsClean execution_memory
#axiomsClean quantized_defined
#axiomsClean value_readback
#axiomsClean scale_readback
#axiomsClean frame
#axiomsClean written_policy_congr
#axiomsClean policy_independent
#axiomsClean residual_backend_independent
end
end VeriTile.Bench.TritonBenchG.QuantizeCopyKv.TypedQuantizer
