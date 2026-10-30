import bench.tritonbench_g.int8_matmul_quantization.Source
import bench.tritonbench_g.int8_matmul_quantization.Proofs.KernelProducerTargets
import VeriTile.Triton.Float.ScalarOpsLaws

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Producer
open VeriTile.Triton FP.Structural
open KernelProducerGrid
open KernelProducerTargets (Selection Reads nativeMaximum nativeScale nativeByte nativeStoredScale)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩
abbrev Selection := KernelProducerTargets.Selection
abbrev Schedule := KernelProducerGrid.Schedule

theorem native_loaded (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Selection p) (rest : Algebra Word32) (values : Nat → Nat → Word32)
    (s : State Word32) (reads : Reads p values s) (r : Fin p.rows) (c : Fin p.width) :
    KernelProducer.loaded (KernelScalarProfile.native f selection.tree rest) (parameters p)
      (atRow s r) (0,c,PUnit.unit) =
      if c.val < p.columns then values r c else FP.Value.zero .fp32 := by
  by_cases hc : c.val < p.columns
  · simp [KernelProducer.loaded, KernelProducer.inputOffset, KernelScalarProfile.native,
      parameters, atRow, Nat.mod_eq_of_lt r.isLt, hc, reads r c r.isLt hc]
  · simp [KernelProducer.loaded, KernelProducer.inputOffset, KernelScalarProfile.native, parameters, hc]

theorem native_maximum (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Selection p) (covers : p.columns ≤ p.width) (rest : Algebra Word32)
    (values : Nat → Nat → Word32) (s : State Word32) (reads : Reads p values s) (r : Fin p.rows) :
    KernelProducer.maximum (KernelScalarProfile.native f selection.tree rest) (parameters p) (atRow s r) =
      nativeMaximum f p selection (values r) := by
  simp only [KernelProducer.maximum, KernelScalarProfile.native, nativeMaximum,
    KernelScalarProfile.reduceTree]
  simp only [ite_true]
  apply congrArg (f.maximum (FP.Value.zero .fp32))
  apply congrArg (fun leaves => selection.tree.evaluate f.maximum leaves)
  funext i
  by_cases hi : i < p.width
  · have loaded := congrArg FP.Value.abs (native_loaded f p selection rest values s reads r ⟨i, hi⟩)
    have zero := FP.ScalarOpsLaws.abs_positive_zero_fp32
    by_cases hc : i < p.columns
    · simpa [parameters, TileShape.axisDim, hi, TileShape.insertAxisIndex, hc] using loaded
    · simpa [parameters, TileShape.axisDim, hi, TileShape.insertAxisIndex, hc, zero] using loaded
  · have hc : ¬ i < p.columns := by omega
    simp [parameters, TileShape.axisDim, hi, hc]

theorem native_scale (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Selection p) (covers : p.columns ≤ p.width) (rest : Algebra Word32)
    (values : Nat → Nat → Word32) (s : State Word32) (reads : Reads p values s) (r : Fin p.rows) :
    KernelProducer.scale (KernelScalarProfile.native f selection.tree rest) (parameters p) (atRow s r) =
      nativeScale f p selection (values r) := by
  unfold KernelProducer.scale nativeScale
  rw [native_maximum f p selection covers rest values s reads r]
  simp [KernelScalarProfile.native]

theorem native_quotient (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Selection p) (covers : p.columns ≤ p.width) (rest : Algebra Word32)
    (values : Nat → Nat → Word32) (s : State Word32) (reads : Reads p values s)
    (r : Fin p.rows) (c : Fin p.width) (hc : c.val < p.columns) :
    KernelNativeModel.integer8
      (scaled (KernelScalarProfile.native f selection.tree rest) p s r c) =
      nativeByte f p selection (values r) c := by
  unfold scaled KernelProducer.quotient nativeByte
  rw [native_loaded f p selection rest values s reads r c, if_pos hc,
    native_scale f p selection covers rest values s reads r]
  rfl

theorem native_stored_scale (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Selection p) (covers : p.columns ≤ p.width) (rest : Algebra Word32)
    (values : Nat → Nat → Word32) (s : State Word32) (reads : Reads p values s) (r : Fin p.rows) :
    scale (KernelScalarProfile.native f selection.tree rest) p s r =
      KernelScalarProfile.pack16 (nativeStoredScale f p selection (values r)) := by
  unfold scale nativeStoredScale
  rw [native_scale f p selection covers rest values s reads r]
  rfl

/-- Exact native bytes/scales from the complete quoted producer grid. The
divide/max policy remains local; the result contains no ideal=FP assertion. -/
theorem native_complete (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Selection p) (covers : p.columns ≤ p.width) (rest : Algebra Word32)
    (values : Nat → Nat → Word32) (s : State Word32) (reads : Reads p values s) (order : Schedule p) :
    ∃ final, KernelProducerGrid.run (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8 p order.order s = some final ∧
      (∀ r : Fin p.rows, ∀ c : Fin p.width, c.val < p.columns →
        final.mem "q" (r.val*p.columns+c.val) = .mk .int (nativeByte f p selection (values r) c).toInt) ∧
      (∀ r : Fin p.rows, final.mem "scales" r =
        .mk .fp16 (KernelScalarProfile.pack16 (nativeStoredScale f p selection (values r)))) ∧
      (∀ (region : RegionName) offset, (region ≠ "q" ∨ p.rows*p.columns ≤ offset) →
        (region ≠ "scales" ∨ p.rows ≤ offset) → final.mem region offset = s.mem region offset) := by
  obtain ⟨final, he, hq, hs, hf⟩ := source_correct (KernelScalarProfile.native f selection.tree rest)
    KernelNativeModel.integer8 p order s
  refine ⟨final, he, ?_, ?_, hf⟩
  · intro r c hc
    rw [hq r c hc, native_quotient f p selection covers rest values s reads r c hc]
  · intro r
    rw [hs r, native_stored_scale f p selection covers rest values s reads r]

#axiomsClean native_loaded
#axiomsClean native_maximum
#axiomsClean native_scale
#axiomsClean native_complete

theorem mathematical_loaded (p : Geometry) (selection : Selection p) (rest : Algebra ℝ)
    (values : Nat → Nat → ℝ) (s : State ℝ) (reads : Reads p values s)
    (r : Fin p.rows) (c : Fin p.width) :
    KernelProducer.loaded (KernelScalarProfile.mathematical selection.tree rest) (parameters p)
      (atRow s r) (0,c,PUnit.unit) = if c.val < p.columns then values r c else 0 := by
  by_cases hc : c.val < p.columns
  · simp [KernelProducer.loaded, KernelProducer.inputOffset, KernelScalarProfile.mathematical,
      parameters, atRow, Nat.mod_eq_of_lt r.isLt, hc, reads r c r.isLt hc]
  · simp [KernelProducer.loaded, KernelScalarProfile.mathematical, parameters, hc]

theorem mathematical_maximum (p : Geometry) (selection : Selection p) (covers : p.columns ≤ p.width)
    (rest : Algebra ℝ) (values : Nat → Nat → ℝ) (s : State ℝ) (reads : Reads p values s) (r : Fin p.rows) :
    KernelProducer.maximum (KernelScalarProfile.mathematical selection.tree rest) (parameters p) (atRow s r) =
      maximumAbs (values r) p.columns := by
  rw [← KernelProducerTargets.mathematical_maximum p selection (values r) covers]
  simp only [KernelProducer.maximum, KernelScalarProfile.mathematical, KernelScalarProfile.reduceTree]
  apply congrArg (max 0)
  apply congrArg (fun leaves => selection.tree.evaluate max leaves)
  funext i
  by_cases hi : i < p.width
  · have loaded := congrArg abs (mathematical_loaded p selection rest values s reads r ⟨i,hi⟩)
    by_cases hc : i < p.columns <;>
      simpa [parameters, TileShape.axisDim, hi, TileShape.insertAxisIndex, hc] using loaded
  · have hc : ¬ i < p.columns := by omega
    simp [parameters, TileShape.axisDim, hi, hc]

theorem mathematical_scale (p : Geometry) (selection : Selection p) (covers : p.columns ≤ p.width)
    (rest : Algebra ℝ) (values : Nat → Nat → ℝ) (s : State ℝ) (reads : Reads p values s) (r : Fin p.rows) :
    KernelProducer.scale (KernelScalarProfile.mathematical selection.tree rest) (parameters p) (atRow s r) =
      KernelProducerTargets.mathematicalScale (values r) p.columns := by
  unfold KernelProducer.scale KernelProducerTargets.mathematicalScale
  rw [mathematical_maximum p selection covers rest values s reads r]
  rfl

/-- Separate ideal interpretation. Floating casts are interpreted as Real
identity here; this theorem makes no native word-to-Real equality claim. -/
theorem mathematical_complete (p : Geometry) (selection : Selection p) (covers : p.columns ≤ p.width)
    (rest : Algebra ℝ) (values : Nat → Nat → ℝ) (s : State ℝ) (reads : Reads p values s) (order : Schedule p) :
    ∃ final, KernelProducerGrid.run (KernelScalarProfile.mathematical selection.tree rest) realInteger8 p order.order s = some final ∧
      (∀ r : Fin p.rows, ∀ c : Fin p.width, c.val < p.columns →
        final.mem "q" (r.val*p.columns+c.val) =
          .mk .int (KernelProducerTargets.mathematicalByte (values r) p.columns c).toInt) ∧
      (∀ r : Fin p.rows, final.mem "scales" r =
        .mk .fp16 (KernelProducerTargets.mathematicalScale (values r) p.columns)) ∧
      (∀ (region : RegionName) offset, (region ≠ "q" ∨ p.rows*p.columns ≤ offset) →
        (region ≠ "scales" ∨ p.rows ≤ offset) → final.mem region offset = s.mem region offset) := by
  obtain ⟨final, he, hq, hs, hf⟩ := source_correct (KernelScalarProfile.mathematical selection.tree rest)
    realInteger8 p order s
  refine ⟨final, he, ?_, ?_, hf⟩
  · intro r c hc
    rw [hq r c hc]
    unfold scaled KernelProducer.quotient KernelProducerTargets.mathematicalByte
    rw [mathematical_loaded p selection rest values s reads r c, if_pos hc,
      mathematical_scale p selection covers rest values s reads r]
    rfl
  · intro r
    rw [hs r]
    unfold scale
    rw [mathematical_scale p selection covers rest values s reads r]
    rfl

#axiomsClean mathematical_maximum
#axiomsClean mathematical_complete
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Producer
