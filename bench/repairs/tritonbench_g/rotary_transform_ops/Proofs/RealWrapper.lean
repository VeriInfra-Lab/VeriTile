import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.Binding

namespace VeriTile.Bench.Repairs.RotaryTransformOps
open Checked Physical Host

/-- Full source wrapper over independent arbitrary real payload/coefficient
memory. Only integer metadata is read from the separate byte observations. -/
theorem whole_arbitrary_real (call : Host.Call) (metadata : Bytes) (domain : Admitted call metadata)
    (callerDevice : Nat) (m : Memory ℝ) :
    RealClaim call callerDevice metadata m (mathematical call callerDevice metadata m) := by
  refine ⟨rfl,rfl,rfl,rfl,⟨rfl,rfl⟩,mathematical_memory call callerDevice metadata m domain.config,?_,?_⟩
  · exact correct_real call callerDevice metadata m domain.config
  · intro q a noCopy outside
    rw [mathematical_memory call callerDevice metadata m domain.config,source_eq_run _ domain.config,
      run_frame _ realOps _ q a outside]
    simp [prepare,noCopy]

#axiomsClean whole_arbitrary_real
end VeriTile.Bench.Repairs.RotaryTransformOps
