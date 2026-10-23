import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.CheckedHost
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Native
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.NativeLaws

namespace VeriTile.Bench.Repairs.LayerNormOps.HostFPEquiv
open VeriTile.Triton FP.Structural Admission Checked ScalarPlan
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section

/-- Complete bit-expression evaluation, including both grids and native host
sums/casts. Scalar/configuration ports and complete local reduction/contraction
plans remain the explicit numerical/compiler boundary, never result oracles. -/
theorem layer_norm_ops_whole_fp32 (P : Primitives) (contract : Contraction) (trees : Trees)
    (f : ForwardCall) (b : BackwardCall) (fr : ForwardParameters) (br : BackwardParameters)
    (hf : ForwardContract f) (hb : BackwardContract b) (binding : CompositionBinding f b fr br)
    (pw pb : ReductionPlan.Plan b.programs) (dw db : RegionName) (s : State Expr)
    (live : List RegionName)
    (fresh : Scope.Allocations (forwardParameters f fr) (backwardParameters b br) dw db live) :
    Scope.Supported (forwardParameters f fr) (backwardParameters b br) f.common.rows b.programs ∧
    (Checked.composed (algebra contract trees) (hostScalar contract) f b fr br pw pb dw db s).map
      (Option.map (Native.observed P)) =
      (let first := Host.forwardFinish (forwardParameters f fr) f.common.rows
        (Host.Grid.reference (Forward.reference (algebra contract trees) (forwardParameters f fr)) s f.common.rows)
       .ok (some (Native.observed P (first,
         Host.backwardFinish (hostScalar contract) (backwardParameters b br) b.programs pw pb dw db
           f.residual.present (Host.Grid.reference
             (Backward.reference (algebra contract trees) (backwardParameters b br)) first.state b.programs))))) ∧
    (∃ result, Checked.composed (algebra contract trees) (hostScalar contract) f b fr br pw pb dw db s = .ok (some result) ∧
      (∀ r, r ∈ live → ∀ offset, evalCell P (result.2.state.mem r offset) = evalCell P (s.mem r offset))) := by
  refine ⟨Checked.supported f b fr br hf hb binding, ?_, ?_⟩
  · rw [Checked.composed_admitted _ _ _ _ _ _ hf hb binding, Host.composed_complete]
    rfl
  · obtain ⟨result, hr, frame⟩ := Checked.caller_frame (algebra contract trees) (hostScalar contract)
      f b fr br hf hb binding pw pb dw db s live fresh
    exact ⟨result, hr, fun r hr offset => congrArg (evalCell P) (frame r hr offset)⟩

#axiomsClean layer_norm_ops_whole_fp32
end
end VeriTile.Bench.Repairs.LayerNormOps.HostFPEquiv
