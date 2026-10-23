import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelWhole
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelEncoding

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelRealResult
open VeriTile.Triton FP.Structural
noncomputable section

/-- Real is the mathematical interpretation. Floating storage casts are erased
here; signed64 register interpretation and final byte encoding stay explicit. -/
def ops : Operations ℝ := idealOperations (fun _ => ⟨0, by decide⟩)
def round (x : ℝ) : BitVec 64 := BitVec.ofInt 64 (roundEven x)
def algebra (p : Launch) (rest : Algebra ℝ) : Algebra ℝ :=
  KernelProfile.algebra ops id p.tree rest

/-- Independent Tensor formula, covering every active row/column and the
complete exterior logical-memory frame. -/
def Result (p : Launch) (values : Nat → Nat → ℝ) (initial final : State ℝ) : Prop :=
  (∀ r c, r < p.rows → c < p.columns →
    narrow8 ((final.mem "q" (r*p.columns+c)).read .int) = mathematicalByte (values r) p.columns c) ∧
  (∀ r, r < p.rows → (final.mem "m" r).read .fp16 = logicalMaximum (values r) p.columns) ∧
  (∀ (region : RegionName) offset, (region ≠ "q" ∨ p.rows*p.columns ≤ offset) →
    (region ≠ "m" ∨ p.rows ≤ offset) → final.mem region offset = initial.mem region offset)

theorem kernel_correct (rest : Algebra ℝ) (p : Launch) (schedule : Schedule p)
    (values : Nat → Nat → ℝ) (initial : State ℝ) (reads : KernelWhole.Reads p values initial) :
    ∃ final, KernelGrid.run (algebra p rest) round (KernelProfile.equal ops)
      p schedule.order initial = some final ∧ Result p values initial final := by
  obtain ⟨final, run, hq, hm, hf⟩ := KernelWhole.source_correct ops id round rest p schedule values initial reads
  refine ⟨final, run, ?_, ?_, hf⟩
  · intro r c hr hc
    rw [hq r c hr hc]
    change narrow8 (KernelExecution.signedByte (round
      (rowScaled ops (values r c) (rowMaximum ops p (values r))))) = _
    rw [KernelExecution.narrow_signedByte]
    change narrow8 (BitVec.ofInt 64 (roundEven
      (rowScaled ops (values r c) (rowMaximum ops p (values r))))).toInt = _
    rw [KernelEncoding.narrow64]
    exact source_quantization_is_logical (fun _ => ⟨0, by decide⟩) p (values r) c hc
  · intro r hr
    rw [hm r hr]
    exact source_maximum_is_logical (fun _ => ⟨0, by decide⟩) p (values r)

#axiomsClean kernel_correct
#specNonCircular Result avoiding [RowwiseQuantizationTriton.source, KernelExecution.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelRealResult
