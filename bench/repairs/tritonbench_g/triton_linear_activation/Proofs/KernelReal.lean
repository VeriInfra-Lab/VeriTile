import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelMemory
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelContract
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelRealArithmetic

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelReal
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelContract KernelRealArithmetic
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Every active output and saved preactivation of the actual source, plus
its entire tagged-cell frame. Store ownership follows from checked canonical
strides; successful execution and completed output are conclusions. -/
theorem outputs (mode : Mode) (p : Parameters) (s : State ℝ)
    (positive : 0<p.blockInner) (strides : CanonicalStrides p) (separate : p.saved≠p.output) :
    ∃ final, exec (engine algebra dot) (source mode p) s=some final ∧
      (∀ i, active p s i=Bool.true → final.mem p.output (KernelValues.outputOffset p s i)=
        outputCell algebra mode (denseOutput mode p s (KernelValues.row p s i.1) (KernelValues.col p s i.2.1))) ∧
      (p.saveInputs=Bool.true → ∀ i, active p s i=Bool.true → final.mem p.saved (savedOffset p s i)=
        outputCell algebra mode (densePre mode p s (KernelValues.row p s i.1) (KernelValues.col p s i.2.1))) ∧
      (∀ region offset,
        (region≠p.output ∨ ∀ i, active p s i=Bool.true → offset≠KernelValues.outputOffset p s i) →
        (p.saveInputs=Bool.false ∨ region≠p.saved ∨ ∀ i, active p s i=Bool.true → offset≠savedOffset p s i) →
        final.mem region offset=s.mem region offset) := by
  obtain ⟨final,execution,out,saved,frame⟩ := KernelMemory.outputs algebra dot mode p s positive separate
    (output_unique p s strides) (saved_unique p s strides)
  refine ⟨final,execution,?_,?_,frame⟩
  · intro i enabled
    have bounds : KernelValues.row p s i.1<p.rows ∧ KernelValues.col p s i.2.1<p.columns := by
      simpa [active] using enabled
    rw [out i enabled,accumulated_lane mode p s positive i bounds.1 bounds.2,KernelRealArithmetic.activation_real]
    rfl
  · intro flag i enabled
    have bounds : KernelValues.row p s i.1<p.rows ∧ KernelValues.col p s i.2.1<p.columns := by
      simpa [active] using enabled
    rw [saved flag i enabled,accumulated_lane mode p s positive i bounds.1 bounds.2]

#axiomsClean outputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelReal
