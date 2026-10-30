import bench.repairs.tritonbench_g.fused_layernorm_triton.Programs
import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.PairIO
import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.IOResults
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Pair
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000

private theorem mapCell_comp {α β γ : Type} (f : α → β) (g : β → γ) (c : Cell α) :
    mapCell (g ∘ f) c=mapCell g (mapCell f c) := by
  cases c with
  | mk d v => cases d <;> rfl

theorem implements {p : Parameters} (program : Program p) (I : Native.Instructions) (P : Native.Profile) :
    (program.io I P).Implements
      (PairIO.expected p.X p.W p.Bias p.Y p.Mean p.Rstd p.rows p.D
        (IOResults.nativeFormula I P p.D p.configuration)) := by
  apply PairIO.implements
  intro initial xs ws bs hx hw hb
  obtain ⟨final,ran,values,frame⟩ := KernelValues.nativeSource program.hoist I P initial
    p.X p.W p.Bias p.Y p.Mean p.Rstd p.rows p.D xs ws bs hx hw hb p.separation p.configuration
  refine ⟨final,ran,?_,frame⟩
  intro row hr
  obtain ⟨hm,hrstd,hy⟩ := values row hr
  refine ⟨?_,?_,?_⟩
  · rw [mapCell_comp,show mapCell (Native.observe I P) (final.mem p.Mean row)=_ from hm];rfl
  · rw [mapCell_comp,show mapCell (Native.observe I P) (final.mem p.Rstd row)=_ from hrstd];rfl
  · intro col hc
    rw [mapCell_comp,show mapCell (Native.observe I P) (final.mem p.Y (col+p.D*row))=_ from hy col hc];rfl

/-- The only transformation is moving the three invariant scalar projections
before the normalization loop. Every FP operation and selected tree is retained. -/
theorem equivalent (p : Parameters) : original p ≡[assumptions] candidate p := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent)
    (lhs := original p) (rhs := candidate p) rfl rfl
  intro I P
  exact ProgramIO.EquivalentOn.of_implementsOn (implements (original p) I P) (implements (candidate p) I P)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Pair
