import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety00
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety01
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety10
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety11

namespace VeriTile.Bench.Repairs.LayerNormOps.SourceSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward (M : Algebra α) (p : ForwardParameters) (s : State α)
    (bounds : RegionName → Nat)
    (hx : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.xStride+i.val<bounds p.x)
    (hw : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.weight)
    (hb : p.hasBias=Bool.true → ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.bias)
    (hres : p.hasResidual=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      s.pids 0*p.residualStride+i.val<bounds p.residual)
    (hy : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.yStride+i.val<bounds p.y)
    (hresout : p.storeResidual=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      s.pids 0*p.residualOutStride+i.val<bounds p.residualOut)
    (hm : p.rms=Bool.false → s.pids 0<bounds p.mean)
    (hr : s.pids 0<bounds p.rstd) :
    TraceSafe M bounds (forwardKernel p) s := by
  cases hR : p.rms <;> cases hA : p.hasResidual
  · exact forward00 M p s bounds hx hw hb hres hy hresout hm hr hR hA
  · exact forward01 M p s bounds hx hw hb hres hy hresout hm hr hR hA
  · exact forward10 M p s bounds hx hw hb hres hy hresout hm hr hR hA
  · exact forward11 M p s bounds hx hw hb hres hy hresout hm hr hR hA

#axiomsClean forward
end
end VeriTile.Bench.Repairs.LayerNormOps.SourceSafety
