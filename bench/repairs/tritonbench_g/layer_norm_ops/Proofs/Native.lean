import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Host
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ScalarPlan
namespace VeriTile.Bench.Repairs.LayerNormOps.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
open ScalarPlan
noncomputable section

/-- Whole source, both grids and host axis reductions, retaining permitted
local contraction and complete ordered reduction trees. -/
theorem complete (contract : Contraction) (trees : Trees)
    (fp : ForwardParameters) (bp : BackwardParameters) (rows programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (s : State Expr) :
    Host.composed (algebra contract trees) (hostScalar contract) fp bp rows programs pw pb dw db s =
      let f := Host.forwardFinish fp rows
        (Host.Grid.reference (Forward.reference (algebra contract trees) fp) s rows)
      some (f,Host.backwardFinish (hostScalar contract) bp programs pw pb dw db fp.hasResidual
        (Host.Grid.reference (Backward.reference (algebra contract trees) bp) f.state programs)) :=
  Host.composed_complete _ _ _ _ _ _ _ _ _ _ _

def observed (P : Primitives) (result : Host.ForwardReturn Expr × Host.BackwardReturn Expr) :=
  (result.1.y,result.1.mean,result.1.rstd,result.1.z,
   result.2.dx,result.2.dw,result.2.db,result.2.dResidual,result.2.recomputedY,
   fun r o => evalCell P (result.2.state.mem r o))

/-- Concrete software-FP results evaluate only scalar nodes in the actual
returned memory. No result buffer, row or kernel is a primitive argument. -/
theorem complete_bits (P : Primitives) (contract : Contraction) (trees : Trees)
    (fp : ForwardParameters) (bp : BackwardParameters) (rows programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (s : State Expr) :
    (Host.composed (algebra contract trees) (hostScalar contract) fp bp rows programs pw pb dw db s).map
      (observed P) =
      let f := Host.forwardFinish fp rows
        (Host.Grid.reference (Forward.reference (algebra contract trees) fp) s rows)
      some (observed P (f,Host.backwardFinish (hostScalar contract) bp programs pw pb dw db fp.hasResidual
        (Host.Grid.reference (Backward.reference (algebra contract trees) bp) f.state programs))) := by
  rw [complete]
  rfl

#axiomsClean complete
#axiomsClean complete_bits
#specNonCircular ScalarPlan.Expr.eval avoiding [forwardKernel,backwardKernel,Forward.reference,Backward.reference]
end
end VeriTile.Bench.Repairs.LayerNormOps.Native
