import bench.tritonbench_g.quantize_kv_transform.Proofs.TypedSource
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.TritonBenchG.QuantizeKvTransform.TypedTransform
open VeriTile.Triton FP.Structural FP.IntegerConversion
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

theorem conversion_defined (u : Word → Int) (x : Word) (h : (toSigned 8 x).isSome) :
    totalSigned u 8 x = (toSigned 8 x).getD 0 := by
  cases hx : toSigned 8 x with
  | none => simp [hx] at h
  | some z => simp [totalSigned, hx]

attribute [local irreducible] MixedBits.narrow MixedBits.widen MixedBits.absHalf
  FP.div QuantizeKvCopy.TypedQuantize.maximum toSigned totalSigned

theorem execution_memory (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (d : Storage) (p : Parameters) (s : State Word) (hw : 0 < p.width)
    (h : InputDefined cfg d p s) :
    (execution rest cfg u d p s hw).val.mem = (written cfg d p s).mem := by
  cases d
  all_goals
    simp +contextual [execution, written, valuesWritten, masked_scatter_memory,
      active, inputOffset, outputOffset, scaleOffset, destination, expected,
      loaded, scale, maximum, quotient, promoted, absPromoted, narrow_zero,
      QuantizeKvCopy.TypedQuantize.reduction,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
      TileShape.insertAxisIndex, TileShape.replaceAxisIndex, TileShape.dropInsertedIndex, TileShape.eraseAxis,
      show (0.0 : ℝ) = 0 by norm_num, show (127.0 : ℝ) = 127 by norm_num]
    apply congrArg₂ (fun f mem => List.foldl f mem (TileShape.allIndices [p.height, 1]))
    · funext mem i
      by_cases hi : i.1.val < p.heads
      · simp only [hi, ↓reduceIte]
        funext r o
        split_ifs <;> try rfl
        congr 4
        funext j
        first
        | change MixedBits.widen (MixedBits.absHalf
            (if i.1.val < p.heads ∧ j.val < p.dimension then
              (s.mem p.input (s.pids 0 * p.strideKbs + i.1.val * p.strideKh + p.strideKd * j.val)).read .fp16
            else FP.Value.zero .fp32)) = _
        | change FP.Value.abs
            (if i.1.val < p.heads ∧ j.val < p.dimension then
              (s.mem p.input (s.pids 0 * p.strideKbs + i.1.val * p.strideKh + p.strideKd * j.val)).read .real
            else FP.Value.zero .fp32) = _
        simp [hi]
      · simp only [hi, ↓reduceIte]
    apply congrArg (fun f => List.foldl f s.mem (TileShape.allIndices [p.height, p.width]))
    funext mem i
    by_cases hi : active p i
    · have hc := conversion_defined (u 8) _ (h i hi)
      simp only [active] at hi
      simp [quotient, scale, maximum, loaded, inputOffset, active, promoted, absPromoted, hi] at hc
      simp only [hi, ↓reduceIte]
      funext r o
      change (if r = p.output ∧ o = outputOffset p s i then _ else mem r o) =
        (if r = p.output ∧ o = outputOffset p s i then _ else mem r o)
      split_ifs with ho
      · apply congrArg (Cell.mk .int)
        convert hc using 1 <;> congr 3
        congr 3
        funext j
        first
        | change MixedBits.widen (MixedBits.absHalf
            (if i.1.val < p.heads ∧ j.val < p.dimension then
              (s.mem p.input (s.pids 0 * p.strideKbs + i.1.val * p.strideKh + p.strideKd * j.val)).read .fp16
            else FP.Value.zero .fp32)) = _
        | change FP.Value.abs
            (if i.1.val < p.heads ∧ j.val < p.dimension then
              (s.mem p.input (s.pids 0 * p.strideKbs + i.1.val * p.strideKh + p.strideKd * j.val)).read .real
            else FP.Value.zero .fp32) = _
        simp [hi]
      · rfl
    · simp only [active] at hi
      simp only [hi, ↓reduceIte]

theorem maximum_spec (d : Storage) (p : Parameters) (s : State Word)
    (g : Fin p.height) (hw : 0 < p.width) :
    MatrixReduction.MachineMax.IsMaximum .fp32
      (fun j : Fin p.width => (absPromoted d (loaded d p s (g, j, PUnit.unit))).bits)
      (maximum d p s g).bits := by
  simpa [maximum, QuantizeKvCopy.TypedQuantize.maximum,
    MatrixReduction.MachineMax.rowMaximum, hw] using
    MatrixReduction.MachineMax.best_spec .fp32
      (fun j : Fin p.width => (absPromoted d (loaded d p s (g, j, PUnit.unit))).bits) hw

theorem expected_spec (cfg : FP.Config) (d : Storage) (p : Parameters) (s : State Word)
    (h : InputDefined cfg d p s) (i : TileIndex [p.height, p.width]) (hi : active p i) :
    ∃ q, (quotient cfg d p s i).toRat? = some q ∧
      TowardZero q (expected cfg d p s i) ∧ SignedRange 8 (expected cfg d p s i) := by
  have hd := h i hi
  cases hx : toSigned 8 (quotient cfg d p s i) with
  | none => simp [hx] at hd
  | some z => simpa [expected, hx] using (toSigned_spec 8 (quotient cfg d p s i) z).mp hx

theorem value_readback (cfg : FP.Config) (d : Storage) (p : Parameters) (s : State Word)
    (i : TileIndex [p.height, p.width]) (hi : active p i)
    (hregions : p.output ≠ p.scales)
    (hinj : ∀ a b, active p a → active p b → outputOffset p s a = outputOffset p s b → a = b) :
    (written cfg d p s).mem p.output (outputOffset p s i) = .mk .int (expected cfg d p s i) := by
  unfold written
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hregions)]
  exact masked_readback_active _ _ _ _ _ i hi hinj

theorem scale_readback (cfg : FP.Config) (d : Storage) (p : Parameters) (s : State Word)
    (i : TileIndex [p.height, 1]) (hi : i.1.val < p.heads)
    (hinj : ∀ a b : TileIndex [p.height, 1], a.1.val < p.heads → b.1.val < p.heads →
      scaleOffset p s a = scaleOffset p s b → a = b) :
    (written cfg d p s).mem p.scales (scaleOffset p s i) = .mk .fp16 (scale cfg d p s i.1) := by
  exact masked_readback_active _ _ _ _ _ i hi hinj

theorem frame (cfg : FP.Config) (d : Storage) (p : Parameters) (s : State Word) (r : RegionName) (o : Nat)
    (hv : r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width], active p i → o ≠ outputOffset p s i)
    (hs : r ≠ p.scales ∨ ∀ i : TileIndex [p.height, 1], i.1.val < p.heads → o ≠ scaleOffset p s i) :
    (written cfg d p s).mem r o = s.mem r o := by
  unfold written valuesWritten
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (hs.imp id (fun h i _ => h i))]
  exact State.masked_scatter_frame _ _ _ _ _ _ _ (hv.imp id (fun h i _ => h i)) _

theorem undefined_policy_independent (rest : Algebra Word) (cfg : FP.Config)
    (u v : Nat → Word → Int) (d : Storage) (p : Parameters) (s : State Word)
    (hw : 0 < p.width) (h : InputDefined cfg d p s) :
    (execution rest cfg u d p s hw).val.mem = (execution rest cfg v d p s hw).val.mem := by
  rw [execution_memory rest cfg u d p s hw h, execution_memory rest cfg v d p s hw h]

/-- Both supplied source formats, actual fp16 scale storage/reload conversion,
independent encoded maximum and exact quotient truncation, both readbacks and
a frame for every address outside the two original active write sets. -/
theorem correctness (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (d : Storage) (p : Parameters) (s : State Word) (hw : 0 < p.width) (h : InputDefined cfg d p s)
    (hregions : p.output ≠ p.scales)
    (hinj : ∀ a b, active p a → active p b → outputOffset p s a = outputOffset p s b → a = b)
    (hscale : ∀ a b : TileIndex [p.height, 1], a.1.val < p.heads → b.1.val < p.heads →
      scaleOffset p s a = scaleOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (MixedBits.backend rest cfg u) (source d p) s = some t ∧
      (∀ i : TileIndex [p.height, p.width], active p i →
        t.mem p.output (outputOffset p s i) = .mk .int (expected cfg d p s i) ∧
        ∃ q, (quotient cfg d p s i).toRat? = some q ∧
          TowardZero q (expected cfg d p s i) ∧ SignedRange 8 (expected cfg d p s i)) ∧
      (∀ i : TileIndex [p.height, 1], i.1.val < p.heads →
        t.mem p.scales (scaleOffset p s i) = .mk .fp16 (scale cfg d p s i.1) ∧
        MatrixReduction.MachineMax.IsMaximum .fp32
          (fun j : Fin p.width => (absPromoted d (loaded d p s (i.1, j, PUnit.unit))).bits)
          (maximum d p s i.1).bits) ∧
      (∀ r o,
        (r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width], active p i → o ≠ outputOffset p s i) →
        (r ≠ p.scales ∨ ∀ i : TileIndex [p.height, 1], i.1.val < p.heads → o ≠ scaleOffset p s i) →
        t.mem r o = s.mem r o) := by
  refine ⟨(execution rest cfg u d p s hw).val, (execution rest cfg u d p s hw).property, ?_, ?_, ?_⟩
  · intro i hi
    rw [execution_memory rest cfg u d p s hw h]
    exact ⟨value_readback cfg d p s i hi hregions hinj, expected_spec cfg d p s h i hi⟩
  · intro i hi
    rw [execution_memory rest cfg u d p s hw h]
    exact ⟨scale_readback cfg d p s i hi hscale, maximum_spec d p s i.1 hw⟩
  · intro r o hv hs
    rw [execution_memory rest cfg u d p s hw h]
    exact frame cfg d p s r o hv hs

#axiomsClean execution_memory
#axiomsClean maximum_spec
#axiomsClean expected_spec
#axiomsClean value_readback
#axiomsClean scale_readback
#axiomsClean frame
#axiomsClean undefined_policy_independent
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.QuantizeKvTransform.TypedTransform
