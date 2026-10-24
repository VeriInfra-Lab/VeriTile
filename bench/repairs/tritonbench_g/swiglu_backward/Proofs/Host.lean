import bench.repairs.tritonbench_g.swiglu_backward.Proofs.Index
import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.Host

namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
noncomputable section

structure CallMetadata where
  input : VeriTile.Torch.ArgumentMetadata
  upstream : VeriTile.Torch.ArgumentMetadata
  gradient : Option VeriTile.Torch.ArgumentMetadata
  output : Option VeriTile.Torch.ArgumentMetadata
  recompute : Bool

def checkCall (metadata : CallMetadata) : Bool :=
  SwigluFwd.checkPayload metadata.input && SwigluFwd.checkPayload metadata.upstream &&
  (metadata.gradient.map SwigluFwd.checkPayload).getD Bool.true &&
  (if metadata.recompute then (metadata.output.map SwigluFwd.checkPayload).getD Bool.true else Bool.true) &&
  match metadata.input with
  | .nonTensor => Bool.false
  | .tensor t => SwigluFwd.inputShapeOK t.shape

structure CallBinding (metadata : CallMetadata) (call : Call) : Prop where
  input : SwigluFwd.CallBinding ⟨metadata.input,metadata.gradient⟩ call.input call.gradient
  upstream : ∀ t, metadata.upstream = .tensor t → t.shape = call.upstream.shape ∧ t.device.index = call.upstream.cudaDevice
  recompute : metadata.recompute = call.recompute
  outputPresence : metadata.output.isSome = call.output.isSome
  output : ∀ a t, metadata.output = some (.tensor a) → call.output = some t →
    a.shape = t.shape ∧ a.device.index = t.cudaDevice

def checkedPrepare (argsMetadata : CallMetadata) (rt : Runtime α call entry preexisting)
    (storage : ∀ p : Prepared α call entry preexisting, Storage p) :
    Except String (Prepared α call entry preexisting) :=
  if checkCall argsMetadata then
    match rt.prepare with
    | .error reason => .error reason
    | .ok p =>
      if checks p then
        if checkLaunch p (storage p).byteBase then .ok p else .error "launch contract"
      else .ok p
  else .error "payload or shape contract"

def checkedRuntime (argsMetadata : CallMetadata) (rt : Runtime α call entry preexisting)
    (storage : ∀ p : Prepared α call entry preexisting, Storage p) :
    Runtime α call entry preexisting :=
  ⟨checkedPrepare argsMetadata rt storage,rt.setup⟩

theorem prepared_admitted (argsMetadata : CallMetadata) (rt : Runtime α call entry preexisting)
    (storage : ∀ p : Prepared α call entry preexisting, Storage p)
    (p : Prepared α call entry preexisting)
    (hp : (checkedRuntime argsMetadata rt storage).prepare = .ok p) (hc : checks p = Bool.true) :
    checkCall argsMetadata = Bool.true ∧ rt.prepare = .ok p ∧
      LaunchContract p (storage p).byteBase := by
  simp only [checkedRuntime,checkedPrepare] at hp
  split at hp
  · rename_i admitted
    cases original : rt.prepare with
    | error reason => simp [original] at hp
    | ok q =>
      simp only [original] at hp
      split at hp
      · rename_i checked
        split at hp
        · rename_i launch
          cases hp
          exact ⟨admitted,rfl,(checkLaunch_iff _ _).mp launch⟩
        · cases hp
      · rename_i failed
        cases hp
        exact False.elim (failed hc)
  · cases hp

def runtimeDomain (argsMetadata : CallMetadata) (rt : Runtime α call entry preexisting)
    (storage : ∀ p : Prepared α call entry preexisting, Storage p)
    (p : Prepared α call entry preexisting)
    (hp : (checkedRuntime argsMetadata rt storage).prepare = .ok p) (hc : checks p = Bool.true) : Domain p :=
  checkedDomain p (storage p) hc (prepared_admitted argsMetadata rt storage p hp hc).2.2

theorem checked_host_outcomes [Inhabited α] (M : Algebra α) (argsMetadata : CallMetadata)
    (rt : Runtime α call entry preexisting)
    (storage : ∀ p : Prepared α call entry preexisting, Storage p) :
    HostSpec (scalar M) (checkedRuntime argsMetadata rt storage)
      (executeHost M (checkedRuntime argsMetadata rt storage)) :=
  host_outcomes M _ (runtimeDomain argsMetadata rt storage)

#axiomsClean checked_host_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward
