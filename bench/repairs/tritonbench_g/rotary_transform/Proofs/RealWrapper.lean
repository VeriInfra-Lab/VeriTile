import bench.repairs.tritonbench_g.rotary_transform.Proofs.Binding

namespace VeriTile.Bench.Repairs.RotaryTransform.RealWrapper
open Checked Physical Host
noncomputable section

/-- The original coefficient contiguous preparation interpreted over arbitrary
real caller values. Integer metadata remains separately decoded from bytes. -/
def prepared (c : Checked.Call) (m : Memory ℝ) : Memory ℝ := fun region address =>
  if region=.cosine then m (.other 2) (c.views.cosineAt c.host.shape address)
  else if region=.sine then m (.other 3) (c.views.sineAt c.host.shape address)
  else m region address

def invoke (c : Checked.Call) (callerDevice : Nat) (metadata : Bytes) (m : Memory ℝ) : Returned ℝ :=
  mathematical c.host callerDevice (stageInputs c.host.shape c.views c.host.abi.offsets64 metadata) (prepared c m)

structure Claim (c : Checked.Call) (out : Binding.Output c) (callerDevice : Nat)
    (metadata : Bytes) (m : Memory ℝ) (result : Returned ℝ) : Prop where
  returned : result.valueIdentity=out.tensor.object
  inplace : c.host.shape.inplace=Bool.true → result.valueIdentity=c.input.object
  region : result.valueRegion=c.host.shape.output
  shape : result.shape=c.host.shape
  dtype : result.fp32=Bool.true
  device : result.device=c.host.device ∧ result.deviceRestored=callerDevice
  source : result.memory=source (configuration c metadata) realOps (prepare c.host.shape (prepared c m))
  values : ∀ b r h d,b<c.host.shape.batches → r<(configuration c metadata).length b →
    h<c.host.shape.heads → d<c.host.shape.width →
    result.memory c.host.shape.output (c.host.shape.address ((configuration c metadata).start b+r) h d)=
      realFormula (configuration c metadata) (prepared c m) b r h d
  frame : ∀ region address,CallerRegion region →
    (region≠c.host.shape.output ∨ size c.host.shape≤address) → result.memory region address=m region address

theorem memory (c : Checked.Call) (callerDevice : Nat) (metadata : Bytes)
    (accepted : check c metadata=Bool.true) (m : Memory ℝ) :
    (invoke c callerDevice metadata m).memory=
      source (configuration c metadata) realOps (prepare c.host.shape (prepared c m)) := by
  have valid := config_valid c metadata accepted
  have staged : (originalConfiguration c.host.shape c.host.abi
      (stageInputs c.host.shape c.views c.host.abi.offsets64 metadata)).Valid := by
    rw [staged_configuration]
    exact valid
  unfold invoke
  rw [mathematical_memory _ _ _ _ staged,staged_configuration]
  rfl

theorem complete (c : Checked.Call) (out : Binding.Output c) (callerDevice : Nat) (metadata : Bytes)
    (accepted : check c metadata=Bool.true) (m : Memory ℝ) :
    Claim c out callerDevice metadata m (invoke c callerDevice metadata m) := by
  have valid := config_valid c metadata accepted
  have staged : (originalConfiguration c.host.shape c.host.abi
      (stageInputs c.host.shape c.views c.host.abi.offsets64 metadata)).Valid := by
    rw [staged_configuration]
    exact valid
  have identity : c.input.object=c.host.inputIdentity := by
    have facts := (checks c metadata accepted).1.1.2
    simp only [tensorCheck,Bool.and_eq_true,decide_eq_true_eq] at facts
    tauto
  refine ⟨?_,?_,rfl,rfl,rfl,⟨rfl,rfl⟩,memory c callerDevice metadata accepted m,?_,?_⟩
  · change (if c.host.shape.inplace then c.host.inputIdentity else c.host.freshIdentity)=out.tensor.object
    cases hi : c.host.shape.inplace
    · exact (out.freshIdentity hi).1.symm
    · rw [out.inplace hi]
      exact identity.symm
  · intro hi
    change (if c.host.shape.inplace then c.host.inputIdentity else c.host.freshIdentity)=c.input.object
    rw [hi]
    exact identity.symm
  · intro b r h d hb hr hh hd
    have result := correct_real c.host callerDevice
      (stageInputs c.host.shape c.views c.host.abi.offsets64 metadata) (prepared c m) staged b r h d hb
      (by simpa only [staged_configuration] using hr) hh hd
    simpa only [invoke,staged_configuration,Checked.configuration] using result
  · intro region address caller outside
    rw [memory c callerDevice metadata accepted m,source_eq_run _ valid,
      run_frame _ realOps _ region address (Public.outside_writes _ valid region address outside)]
    rw [prepare,if_neg (Public.outside_copy c.host.shape region address outside)]
    simp [prepared,caller.1,caller.2.1]

#axiomsClean complete
#specNonCircular prepared avoiding [source,invoke,Host.mathematical]
end
end VeriTile.Bench.Repairs.RotaryTransform.RealWrapper
