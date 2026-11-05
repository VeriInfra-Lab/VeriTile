import bench.tritonbench_g.fast_rms_layernorm.Proofs.ReaderConnections
import bench.tritonbench_g.fast_rms_layernorm.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Correct
open VeriTile Triton
open scoped BigOperators
open scoped VeriTile.Triton.LogicalIO VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Independent reciprocal RMS over the logical feature count. -/
def inverseRMS (h : Call) (x : Fin h.cols → ℝ) : ℝ :=
  1/Real.sqrt ((∑ j, x j^2)/(h.cols : ℝ)+h.eps)

def effectiveWeight (h : Call) (w : ℝ) : ℝ := if h.gemma then w+1 else w

/-- Complete Y and saved reciprocal RMS in logical row/feature coordinates. -/
def forwardFormula (h : Call) (xs : LogicalIO.Inputs h) : LogicalIO.ForwardOutputs h :=
  ⟨(fun row col => xs.x row col*inverseRMS h (xs.x row)*effectiveWeight h (xs.weight col)),
    fun row => inverseRMS h (xs.x row)⟩

/-- The original saved-r and empty-row initial-DW input conditions. -/
def backwardDomain (h : Call) (xs : LogicalIO.Inputs h) : Prop :=
  (∀ row, xs.savedR row=inverseRMS h (xs.x row)) ∧
  (h.rows=0 → ∀ col, xs.initialDW col=0)

/-- Complete DX and DW; every feature-grid DW launch precedes the DX row grid. -/
def backwardFormula (h : Call) (xs : LogicalIO.Inputs h) : LogicalIO.BackwardOutputs h :=
  ⟨(fun row col =>
      let x := xs.x row
      let g := xs.upstream row
      let r := inverseRMS h x
      r*(g col*effectiveWeight h (xs.weight col))-(r^3/(h.cols : ℝ))*x col*
        (∑ k, g k*effectiveWeight h (xs.weight k)*x k)),
    fun col => ∑ row, xs.upstream row col*xs.x row col*inverseRMS h (xs.x row)⟩

/-- Complete forward grid: independent RMS-scaled Y and saved reciprocal RMS.
The IO declares every finite input/output, actual source launch and exact frame. -/
specification kernel_forward (h : Call) :
    Spec.Real ((LogicalIO.forwardView h) ⊨ (forwardFormula h)) :=
  ReaderConnections.forward h (inverseRMS h) (effectiveWeight h) rfl rfl (forwardFormula h) rfl

/-- Complete weight-gradient grid followed by in-place input gradients. Saved r
has the independently specified forward value. For empty rows the existing
zero-initialized DW is an explicit input and the source schedule is empty. -/
specification kernel_backward (h : Call) :
    Spec.Real ((LogicalIO.backwardView h) ⊨{backwardDomain h}
      (backwardFormula h)) :=
  ReaderConnections.backward h (inverseRMS h) (effectiveWeight h) rfl rfl (backwardFormula h) rfl (backwardDomain h) rfl

#specNonCircular Reference.real avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward]
#specNonCircular Reference.realDomain avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward]
#stmtSurfaceSubset kernel_forward ⊆
  [Ports.Values,Ports.Results,RealIO.program,PreparedAlgorithmProgramIO.Implements,Call,Spec.Real,LogicalIO.forwardView,VeriTile.Triton.LogicalIO.Implements,
   LogicalIO.Inputs,LogicalIO.ForwardOutputs,forwardFormula]
#stmtSurfaceSubset kernel_backward ⊆
  [Ports.Values,Ports.Results,RealIO.program,PreparedAlgorithmProgramIO.ImplementsOn,Call,Spec.Real,LogicalIO.backwardView,VeriTile.Triton.LogicalDomainIO.ImplementsOn,
   LogicalIO.Inputs,LogicalIO.BackwardOutputs,backwardDomain,backwardFormula]
#axiomsClean kernel_forward
#axiomsClean kernel_backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.Correct
