import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Real
import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Host

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding
noncomputable section

structure RealClaim (call : Host.Call) (callerDevice : Nat) (initial : Memory ℝ)
    (result : Host.Returned ℝ) : Prop where
  returnedNone : result.value=none
  identities : result.identities=call.identities
  device : result.deviceRestored=callerDevice
  source : result.memory=actualSource call.config (fun _ => realOps) initial
  values : ∀ kind t h d,t<call.config.shape.tokens → h<call.config.shape.heads kind → d<call.config.shape.width →
    result.memory kind.output (logicalOutput call.config kind t h d)=formula call.config.shape initial kind t h d
  frame : ∀ r a,¬Footprint call.config r a → result.memory r a=initial r a

theorem whole_real (call : Host.Call) (domain : Host.Admitted call)
    (callerDevice : Nat) (initial : Memory ℝ) :
    RealClaim call callerDevice initial (Host.mathematical call callerDevice initial) := by
  have memory := Host.mathematical_memory call domain.config callerDevice initial
  refine ⟨rfl,rfl,rfl,memory,?_,?_⟩
  · rw [memory]
    exact (real_post call.config domain.config initial).1
  · rw [memory]
    exact (real_post call.config domain.config initial).2

#axiomsClean whole_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRotaryEmbedding
