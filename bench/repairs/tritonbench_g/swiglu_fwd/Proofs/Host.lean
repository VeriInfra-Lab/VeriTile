import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.Binding

namespace VeriTile.Bench.Repairs.SwigluFwd
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
noncomputable section

structure CallMetadata where
  input : VeriTile.Torch.ArgumentMetadata
  output : Option VeriTile.Torch.ArgumentMetadata

def checkCall (call : CallMetadata) : Bool :=
  checkPayload call.input && (call.output.map checkPayload).getD Bool.true &&
    match call.input with
    | .nonTensor => Bool.false
    | .tensor t => inputShapeOK t.shape

/-- Query binding for the actual entry tensors. The payload checks below run
before the existing Torch preparation path. -/
structure CallBinding (call : CallMetadata) (input : Tensor) (supplied : Option Tensor) : Prop where
  inputShape : ∀ t, call.input = .tensor t → t.shape = input.shape
  inputDevice : ∀ t, call.input = .tensor t → t.device.index = input.cudaDevice
  outputPresence : call.output.isSome = supplied.isSome
  outputShape : ∀ a t, call.output = some (.tensor a) → supplied = some t → a.shape = t.shape
  outputDevice : ∀ a t, call.output = some (.tensor a) → supplied = some t → a.device.index = t.cudaDevice

def checkedPrepare (call : CallMetadata) (rt : Runtime α input supplied entry preexisting)
    (storage : ∀ p : Prepared α input supplied entry preexisting, Storage p) :
    Except String (Prepared α input supplied entry preexisting) :=
  if checkCall call then
    match rt.prepare with
    | .error reason => .error reason
    | .ok p =>
      if checks p then
        if checkLaunch (metadata p (storage p)) then .ok p else .error "launch contract"
      else .ok p
  else .error "payload or shape contract"

def checkedRuntime (call : CallMetadata) (rt : Runtime α input supplied entry preexisting)
    (storage : ∀ p : Prepared α input supplied entry preexisting, Storage p) :
    Runtime α input supplied entry preexisting :=
  ⟨checkedPrepare call rt storage,rt.setup⟩

theorem prepared_admitted (call : CallMetadata) (rt : Runtime α input supplied entry preexisting)
    (storage : ∀ p : Prepared α input supplied entry preexisting, Storage p)
    (p : Prepared α input supplied entry preexisting)
    (hp : (checkedRuntime call rt storage).prepare = .ok p) (hc : checks p = Bool.true) :
    checkCall call = Bool.true ∧ rt.prepare = .ok p ∧
      LaunchContract (metadata p (storage p)) := by
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
          exact ⟨admitted,rfl,(checkLaunch_iff _).mp launch⟩
        · cases hp
      · rename_i failed
        cases hp
        exact False.elim (failed hc)
  · cases hp

def runtimeDomain (call : CallMetadata) (rt : Runtime α input supplied entry preexisting)
    (storage : ∀ p : Prepared α input supplied entry preexisting, Storage p)
    (p : Prepared α input supplied entry preexisting)
    (hp : (checkedRuntime call rt storage).prepare = .ok p) (hc : checks p = Bool.true) : Domain p :=
  checkedDomain p (storage p) hc (prepared_admitted call rt storage p hp hc).2.2

theorem checked_host_outcomes [Inhabited α] (M : Algebra α) (call : CallMetadata)
    (rt : Runtime α input supplied entry preexisting)
    (storage : ∀ p : Prepared α input supplied entry preexisting, Storage p) :
    HostSpec (scalar M) (checkedRuntime call rt storage)
      (executeHost M (checkedRuntime call rt storage)) :=
  host_outcomes M _ (runtimeDomain call rt storage)

#axiomsClean checked_host_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluFwd
