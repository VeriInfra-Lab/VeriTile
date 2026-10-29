import bench.repairs.tritonbench_g.rotary_emb.Proofs.Host

namespace VeriTile.Bench.Repairs.RotaryEmb
noncomputable section

/-- Entire normal wrapper result, not a single lane or a no-alias conclusion
assumed as a premise. `formula` is independent of every execution definition. -/
structure RealClaim (call : Host.Metadata) (callerDevice : Nat) (initial : Memory ℝ)
    (result : Host.Returned ℝ) : Prop where
  returnedNone : result.value=none
  identities : result.identities=call.identities
  device : result.deviceRestored=callerDevice
  source : result.memory=RotaryEmb.source call.shape realOps initial
  values : ∀ key n h d,n<call.shape.tokens → h<call.shape.heads key → d<call.shape.rotary →
    result.memory (dataRegion key) (call.shape.address key n h d)=formula call.shape initial key n h d
  tail : ∀ key n h d,h<call.shape.heads key → d<call.shape.width → call.shape.rotary≤d →
    result.memory (dataRegion key) (call.shape.address key n h d)=initial (dataRegion key) (call.shape.address key n h d)
  coefficients : ∀ (sine : Bool) a,result.memory (if sine then .sin else .cos) a=initial (if sine then .sin else .cos) a
  frame : ∀ r a,(∀ w,call.shape.Active w → ¬Writes call.shape w r a) → result.memory r a=initial r a

theorem whole_real (call : Host.Metadata) (domain : Host.Admitted call)
    (callerDevice : Nat) (initial : Memory ℝ) :
    RealClaim call callerDevice initial (Host.mathematical call callerDevice initial) := by
  have memory := Host.mathematical_memory call domain.shape callerDevice initial
  refine ⟨rfl,rfl,rfl,memory,?_,?_,?_,?_⟩
  · intro key n h d hn hh hd
    rw [memory]
    exact source_formula call.shape domain.shape initial key n h d hn hh hd
  · intro key n h d hh hd ht
    rw [memory,source_eq_run call.shape domain.shape]
    exact run_tail call.shape domain.shape realOps initial key n h d hh hd ht
  · intro sine a
    rw [memory,source_eq_run call.shape domain.shape]
    exact run_coefficients call.shape realOps initial sine a
  · intro r a outside
    rw [memory,source_eq_run call.shape domain.shape]
    exact run_frame call.shape realOps initial r a outside

#axiomsClean whole_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmb
