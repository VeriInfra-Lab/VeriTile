import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelGrid
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelProfile

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelWhole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2500000
variable {α : Type} [Inhabited α]

/-- Input-only binding. No execution, row result, or output equation occurs. -/
def Reads (p : Launch) (values : Nat → Nat → α) (s : State α) : Prop :=
  ∀ r c, r < p.rows → c < p.columns →
    (s.mem "x" (r*p.columns+c)).read .real = values r c

/-- Full grid output and exterior memory frame for the actual quoted source.
The scalar/native or mathematical specialization is supplied separately. -/
theorem source_correct (ops : Operations α) (half : α → α)
    (round : α → BitVec 64) (rest : Algebra α) (p : Launch) (schedule : Schedule p)
    (values : Nat → Nat → α) (s : State α) (reads : Reads p values s) :
    ∃ final,
      KernelGrid.run (KernelProfile.algebra ops half p.tree rest) round (KernelProfile.equal ops)
        p schedule.order s = some final ∧
      (∀ r c, r < p.rows → c < p.columns →
        final.mem "q" (r*p.columns+c) = .mk .int (KernelExecution.signedByte (round
          (rowScaled ops (values r c) (rowMaximum ops p (values r)))))) ∧
      (∀ r, r < p.rows → final.mem "m" r = .mk .fp16 (half (rowMaximum ops p (values r)))) ∧
      (∀ (region : RegionName) offset, (region ≠ "q" ∨ p.rows*p.columns ≤ offset) →
        (region ≠ "m" ∨ p.rows ≤ offset) → final.mem region offset = s.mem region offset) := by
  let M := KernelProfile.algebra ops half p.tree rest
  let final := KernelGrid.finalState M round (KernelProfile.equal ops) p schedule.order s
  have hmax : ∀ r, r < p.rows →
      rowMaximum ops p (KernelProfile.row (KernelGrid.parameters p) (KernelGrid.atRow s r)) =
        rowMaximum ops p (values r) := by
    intro r hr
    apply rowMaximum_congr
    intro c hc
    simpa [KernelProfile.row, KernelGrid.parameters, KernelGrid.atRow] using reads r c hr hc
  refine ⟨final, KernelGrid.run_final M round (KernelProfile.equal ops) p schedule.order s, ?_, ?_, ?_⟩
  · intro r c hr hc
    let i : TileIndex [p.width] := (⟨c, lt_of_lt_of_le hc p.covers⟩, PUnit.unit)
    have result := KernelGrid.final_quantized M round (KernelProfile.equal ops) p schedule.order s r
      (schedule.complete.mem_iff.mpr (List.mem_range.mpr hr)) i hc
    rw [KernelProfile.scaled_eq ops half rest (KernelGrid.parameters p) p rfl rfl, hmax r hr] at result
    have hv : KernelProfile.row (KernelGrid.parameters p) (KernelGrid.atRow s r) i.1.val = values r c := by
      simpa [KernelProfile.row, KernelGrid.parameters, KernelGrid.atRow, i] using reads r c hr hc
    rw [hv] at result
    exact result
  · intro r hr
    have result := KernelGrid.final_maximum M round (KernelProfile.equal ops) p schedule.order s r
      (schedule.complete.mem_iff.mpr (List.mem_range.mpr hr))
    rw [KernelProfile.stored_eq ops half rest (KernelGrid.parameters p) p rfl rfl, hmax r hr] at result
    exact result
  · intro region offset hq hm
    apply KernelGrid.final_frame_order M round (KernelProfile.equal ops) p schedule.order s region offset
    intro r hr
    have rb := List.mem_range.mp (schedule.complete.mem_iff.mp hr)
    constructor
    · rcases hm with hm | hm
      · exact Or.inl hm
      · exact Or.inr (by simp [KernelGrid.atRow]; omega)
    · rcases hq with hq | hq
      · exact Or.inl hq
      · right
        intro i hi he
        have hb := linear_bound p.rows p.columns r i.1.val rb hi
        simp only [KernelResult.offset, KernelGrid.parameters, KernelGrid.atRow,
          Function.update_self] at he
        omega

#axiomsClean source_correct
#specNonCircular Reads avoiding [RowwiseQuantizationTriton.source, KernelExecution.exec]
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelWhole
