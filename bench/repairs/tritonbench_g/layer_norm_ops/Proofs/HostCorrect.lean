import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.CheckedHost
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalResult
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalHost
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathReturned

/-! Guarded complete forward/backward execution, actual returned-memory
formulas from the pre-call inputs, and the caller-memory frame. -/
namespace VeriTile.Bench.Repairs.LayerNormOps.HostCorrect
open VeriTile.Triton FP.Structural Admission Checked
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section

def realHost : Host.Scalar ℝ := MathReturn.scalar

theorem forward_checks_iff (c : ForwardCall) : forwardCheck c = Bool.true ↔ ForwardContract c :=
  forwardCheck_iff c

theorem backward_checks_iff (c : BackwardCall) : backwardCheck c = Bool.true ↔ BackwardContract c :=
  backwardCheck_iff c

/-- Both checked source calls, complete grids, actual saved-state flow, host
partial sums/casts and every optional return descriptor, with caller frame. -/
theorem layer_norm_ops_whole_real (rest : Algebra ℝ)
    (f : ForwardCall) (b : BackwardCall) (fr : ForwardParameters) (br : BackwardParameters)
    (hf : ForwardContract f) (hb : BackwardContract b) (binding : CompositionBinding f b fr br)
    (pw pb : ReductionPlan.Plan b.programs) (dw db : RegionName) (s : State ℝ)
    (live : List RegionName)
    (fresh : Scope.Allocations (forwardParameters f fr) (backwardParameters b br) dw db live) :
    Scope.Supported (forwardParameters f fr) (backwardParameters b br) f.common.rows b.programs ∧
    ∃ result, Checked.composed (Logical.algebra rest) realHost f b fr br pw pb dw db s = .ok (some result) ∧
      (∀ r, r ∈ live → ∀ offset, result.2.state.mem r offset = s.mem r offset) := by
  exact ⟨Checked.supported f b fr br hf hb binding,
    Checked.caller_frame (Logical.algebra rest) realHost f b fr br hf hb binding pw pb dw db s live fresh⟩

/-- Complete checked calls return the independent normalization and gradient
formulas through their actual tensor descriptors. Freshness binds ordinary
allocations and inert optional names, without any output-value premise. -/
theorem layer_norm_ops_returned_math (rest : Algebra ℝ)
    (f : ForwardCall) (b : BackwardCall) (fr : ForwardParameters) (br : BackwardParameters)
    (hf : ForwardContract f) (hb : BackwardContract b) (binding : CompositionBinding f b fr br)
    (pw pb : ReductionPlan.Plan b.programs) (dw db : RegionName) (s : State ℝ)
    (live : List RegionName)
    (fresh : Scope.Allocations (forwardParameters f fr) (backwardParameters b br) dw db live)
    (regions : MathReturn.RegionBinding (forwardParameters f fr) (backwardParameters b br) dw db) :
    Scope.Supported (forwardParameters f fr) (backwardParameters b br) f.common.rows b.programs ∧
    ∃ result, Checked.composed (Logical.algebra rest) realHost f b fr br pw pb dw db s = .ok (some result) ∧
      MathReturn.ReturnPost (forwardParameters f fr) (backwardParameters b br) s f.common.rows dw db result.1 result.2 ∧
      (∀ r, r∈live → ∀ offset, result.2.state.mem r offset=s.mem r offset) := by
  have scope := Checked.supported f b fr br hf hb binding
  obtain ⟨result,run,post⟩ := MathReturn.complete_math rest (forwardParameters f fr) (backwardParameters b br)
    s f.common.rows b.programs scope pw pb dw db regions
  have checkedRun : Checked.composed (Logical.algebra rest) realHost f b fr br pw pb dw db s=.ok (some result) := by
    rw [Checked.composed_admitted _ _ _ _ _ _ hf hb binding]
    exact congrArg Except.ok run
  refine ⟨scope,result,checkedRun,post,?_⟩
  obtain ⟨other,otherRun,frame⟩ := Checked.caller_frame (Logical.algebra rest) realHost f b fr br hf hb binding pw pb dw db s live fresh
  rw [checkedRun] at otherRun
  cases otherRun
  exact frame

#specNonCircular Logical.y avoiding [forwardKernel, backwardKernel, Host.composed, Checked.composed]
#specNonCircular Logical.dx avoiding [forwardKernel, backwardKernel, Host.composed, Checked.composed]
#axiomsClean Logical.forward_produced
#axiomsClean Logical.backward_dx
#axiomsClean Logical.host_dw
#axiomsClean Logical.host_db
#axiomsClean forward_checks_iff
#axiomsClean backward_checks_iff
#axiomsClean layer_norm_ops_whole_real
#axiomsClean layer_norm_ops_returned_math
end
end VeriTile.Bench.Repairs.LayerNormOps.HostCorrect
