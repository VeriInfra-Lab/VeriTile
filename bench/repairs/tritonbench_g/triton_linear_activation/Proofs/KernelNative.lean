import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelMemory
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelContract
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelNativeModel

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelNative
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelContract KernelNativeModel
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def wordCell (mode : Mode) (x : Word) : Cell Word :=
  .mk (dtype mode).toTileDType (ofFloat (dtype mode) x)

theorem cell_roundtrip (profile : LocalProfile) (policy : CompilerPolicy) (cell : Cell Word) :
    mapCell (evaluate profile policy) (mapCell Term.raw cell)=cell := by
  cases cell with
  | mk d v => cases d <;> rfl

theorem output_cell (profile : LocalProfile) (policy : CompilerPolicy) (mode : Mode) (x : Term) :
    mapCell (evaluate profile policy) (outputCell algebra mode x)=
      wordCell mode (castWord .real (dtype mode) (evaluate profile policy x)) := by
  cases mode <;> rfl

/-- All native output words, optional saved words, and the complete tagged-cell
frame under an explicit compiler policy. Numerical ports are scalar division,
libdevice tanh/erf, and one fixed ISA instruction lane; no arbitrary whole-dot
or output-equivalence hypothesis occurs. -/
theorem outputs (profile : LocalProfile) (policy : CompilerPolicy) (c : Fin 98)
    (mode : Mode) (p : Parameters) (s : State Word)
    (positive : 0<p.blockInner) (strides : CanonicalStrides p) (separate : p.saved≠p.output) :
    ∃ final, execute profile policy c mode p s=some final ∧
      (∀ i, active p s i=Bool.true → final.mem p.output (KernelValues.outputOffset p s i)=
        wordCell mode (resultWord profile policy c mode p s i)) ∧
      (p.saveInputs=Bool.true → ∀ i, active p s i=Bool.true → final.mem p.saved (savedOffset p s i)=
        wordCell mode (savedWord profile policy c mode p s i)) ∧
      (∀ region offset,
        (region≠p.output ∨ ∀ i, active p s i=Bool.true → offset≠KernelValues.outputOffset p s i) →
        (p.saveInputs=Bool.false ∨ region≠p.saved ∨ ∀ i, active p s i=Bool.true → offset≠savedOffset p s i) →
        final.mem region offset=s.mem region offset) := by
  let initial := mapState Term.raw s
  obtain ⟨final,execution,out,saved,frame⟩ := KernelMemory.outputs algebra (dot c mode) mode p initial positive separate
    (output_unique p initial strides) (saved_unique p initial strides)
  refine ⟨mapState (evaluate profile policy) final,?_,?_,?_,?_⟩
  · exact congrArg (Option.map (mapState (evaluate profile policy))) execution
  · intro i enabled
    change mapCell (evaluate profile policy) (final.mem p.output (KernelValues.outputOffset p initial i))=_
    rw [out i enabled,output_cell,accumulated_expression]
    rfl
  · intro flag i enabled
    change mapCell (evaluate profile policy) (final.mem p.saved (savedOffset p initial i))=_
    rw [saved flag i enabled,output_cell,accumulated_expression]
    rfl
  · intro region offset ho hs
    change mapCell (evaluate profile policy) (final.mem region offset)=_
    rw [frame region offset ho hs]
    exact cell_roundtrip profile policy (s.mem region offset)

#axiomsClean outputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelNative
