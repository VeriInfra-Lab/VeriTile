import bench.repairs.tritonbench_g.rotary_emb.Proofs.PublicCorrect
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Binding

namespace VeriTile.Bench.Repairs.RotaryEmb
noncomputable section

/-- Complete ideal-real source execution using the original caller inputs and
the executable contract; no separate shape/ownership premise is requested. -/
theorem rotary_emb_correct {M β : Type} (api : Binding.API M ℝ β)
    (c : Checked.Call) (accepted : Checked.check c=Bool.true) (callerDevice : Nat) (m : M) :
    RealClaim c.metadata callerDevice (Binding.inputs api c m)
      (Host.mathematical c.metadata callerDevice (Binding.inputs api c m)) :=
  whole_real c.metadata (Checked.admitted c accepted) callerDevice (Binding.inputs api c m)

#axiomsClean rotary_emb_correct
end
end VeriTile.Bench.Repairs.RotaryEmb
