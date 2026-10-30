import bench.repairs.tritonbench_g.triton_linear_activation.Contract
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Checked.Bytes

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Checked
noncomputable section
open Classical
set_option maxHeartbeats 8000000

inductive Result (σ β : Type) where
  | ok (value : β) (state : DeviceState σ)
  | error (reason : String) (state : DeviceState σ)

def Result.state : Result σ β → DeviceState σ | .ok _ s => s | .error _ s => s
def Result.bind (r : Result σ β) (next : β → DeviceState σ → Result σ γ) : Result σ γ :=
  match r with | .ok x s => next x s | .error why s => .error why s

theorem bind_ok (r : Result σ β) (next : β → DeviceState σ → Result σ γ) (x : γ) (s : DeviceState σ) :
    r.bind next = .ok x s ↔ ∃ value before, r = .ok value before ∧ next value before = .ok x s := by
  cases r <;> simp [Result.bind]

/-- Ordinary exception locations. The scalar-event fuel describes an actual
interrupted native prefix, not an assumed kernel result or reduction value. -/
structure Failures where
  prepare : Option (Nat×String) := none
  enter : Option (String×Nat) := none
  kernel : Option (Nat×String) := none
  exit : Option (String×Nat) := none
  copyback : Option (Nat×String) := none

def part (abi : LocalEventABI α σ a) (fault : Option (Nat×String)) (code : Code α Unit)
    (s : DeviceState σ) : Result σ Unit :=
  match fault with
  | none => .ok () {s with payload := (code.native abi s.payload).2}
  | some (fuel,why) => .error why {s with payload := runPrefix abi fuel code s.payload}

theorem part_success (abi : LocalEventABI α σ a) (fault : Option (Nat×String)) (code : Code α Unit)
    (before after : DeviceState σ) (h : part abi fault code before = .ok () after) :
    after.payload = (code.native abi before.payload).2 ∧ after.currentDevice = before.currentDevice := by
  cases fault <;> simp [part] at h
  cases h
  exact ⟨rfl,rfl⟩

/-- Launch consumes the entered device state; incorrect ownership is an
ordinary setup error before the source event stream. -/
def launchPart (owner : Nat) (abi : LocalEventABI α σ a) (fault : Option (Nat×String))
    (code : Code α Unit) (s : DeviceState σ) : Result σ Unit :=
  if s.currentDevice = owner then part abi fault code s else .error "launch owner" s

theorem launchPart_success (owner : Nat) (abi : LocalEventABI α σ a) (fault : Option (Nat×String))
    (code : Code α Unit) (before after : DeviceState σ)
    (success : launchPart owner abi fault code before = .ok () after) :
    before.currentDevice = owner ∧
    after.payload = (code.native abi before.payload).2 ∧ after.currentDevice = before.currentDevice := by
  unfold launchPart at success
  split at success
  · exact ⟨‹before.currentDevice = owner›,part_success abi fault code before after success⟩
  · cases success

/-- The entered owner is passed to the source execution. The finally action
runs on both success and failure, and successful exit restores the token. -/
def withOwner (owner : Nat) (failures : Failures) (body : DeviceState σ → Result σ Unit)
    (s : DeviceState σ) : Result σ Unit :=
  match failures.enter with
  | some (why,device) => .error why {s with currentDevice := device}
  | none =>
    let result := body {s with currentDevice := owner}
    match failures.exit with
    | some (why,device) => .error why {result.state with currentDevice := device}
    | none => match result with
      | .ok value after => .ok value {after with currentDevice := s.currentDevice}
      | .error why after => .error why {after with currentDevice := s.currentDevice}

theorem owner_success (owner : Nat) (failures : Failures) (body : DeviceState σ → Result σ Unit)
    (before after : DeviceState σ) (h : withOwner owner failures body before = .ok () after) :
    ∃ inside, body {before with currentDevice := owner} = .ok () inside ∧
      after.payload = inside.payload ∧ after.currentDevice = before.currentDevice := by
  unfold withOwner at h
  cases enter : failures.enter <;> simp only [enter] at h
  · cases exit : failures.exit <;> simp only [exit] at h
    · cases run : body {before with currentDevice := owner} with
      | error why state => simp [run] at h
      | ok value state =>
        cases value
        simp only [run,Result.ok.injEq,true_and] at h
        subst after
        exact ⟨state,rfl,rfl,rfl⟩
    · cases h
  · cases h

/-- Only ordinary metadata operations: reshape receives the actual allocated
output Tensor, and save receives the actual converted operands in source order. -/
structure Tail where
  reshape : Tensor → List Nat → Except String Tensor
  save : Tensor × Option Tensor × Tensor → Except String Unit

def outputTensor (h : Allocation r caller (before : Memory α)) : Tensor :=
  tensor (h.object 0) (h.storage 0) [r.batch.rows,r.n] r.mode r.device

def finish (rt : Tail) (h : Allocation r caller (before : Memory α)) (s : DeviceState σ) : Result σ Tensor :=
  let reshaped := match r.batch with
    | .matrix _ => Except.ok (outputTensor h)
    | .batched _ _ => rt.reshape (outputTensor h) (r.batch.shape r.n)
  match reshaped with
  | .error why => .error why s
  | .ok out => match rt.save h.context with
    | .error why => .error why s
    | .ok () => .ok out s

structure TailContract (rt : Tail) (h : Allocation r caller (before : Memory α)) : Prop where
  reshape : ∀ out, rt.reshape (outputTensor h) (r.batch.shape r.n) = .ok out → out = h.returned

def run (abi : LocalEventABI α σ a) (rt : Tail) (h : Allocation r caller (before : Memory α))
    (failures : Failures) (trials : List (Fin 98)) (c : Fin 98) (s : DeviceState σ) : Result σ Tensor :=
  (part abi failures.prepare h.prepareCode s).bind fun _ prepared =>
    (if r.launch then withOwner r.device failures
      (launchPart r.device abi failures.kernel (h.autotunedKernelCode trials c)) prepared
      else .ok () prepared).bind fun _ launched =>
    (part abi failures.copyback h.copybackCode launched).bind fun _ copied => finish rt h copied

theorem finish_success (rt : Tail) (h : Allocation r caller (before : Memory α)) (law : TailContract rt h)
    (before after : DeviceState σ) (out : Tensor) (success : finish rt h before = .ok out after) :
    out = h.returned ∧ after = before ∧ rt.save h.context = .ok () := by
  unfold finish at success
  have shaped : (match r.batch with
      | .matrix _ => Except.ok (outputTensor h)
      | .batched _ _ => rt.reshape (outputTensor h) (r.batch.shape r.n)) = .ok out ∧
      after = before ∧ rt.save h.context = .ok () := by
    cases es : (match r.batch with
      | .matrix _ => Except.ok (outputTensor h)
      | .batched _ _ => rt.reshape (outputTensor h) (r.batch.shape r.n)) with
    | error why => simp [es] at success
    | ok value =>
      cases ec : rt.save h.context with
      | error why => simp [es,ec] at success
      | ok u =>
        cases u
        simp only [es,ec,Result.ok.injEq] at success
        exact ⟨congrArg Except.ok success.1,success.2.symm,rfl⟩
  refine ⟨?_,shaped.2⟩
  cases batch : r.batch
  · simp only [batch,Except.ok.injEq] at shaped
    rw [←shaped.1]
    simp only [outputTensor,Allocation.returned,batch,Batch.rows,Batch.shape]
  · exact law.reshape out (by simpa [batch] using shaped.1)

theorem native_bind (abi : LocalEventABI α σ a) (code : Code α β) (next : β → Code α γ) (s : σ) :
    (code.bind next).native abi s = (next (code.native abi s).1).native abi (code.native abi s).2 := by
  induction code generalizing s with
  | pure x => rfl
  | event e rest ih =>
    simp only [Code.bind,Code.native]
    exact ih _ _ _

/-- Successful whole calls compose the actual cast/kernel/copyback event
sequence, actual reshape return, actual save order and the CUDA finally path. -/
theorem run_success (abi : LocalEventABI α σ a) (rt : Tail)
    (h : Allocation r caller (before : Memory α)) (law : TailContract rt h)
    (failures : Failures) (trials : List (Fin 98)) (c : Fin 98)
    (initial after : DeviceState σ) (out : Tensor)
    (success : run abi rt h failures trials c initial = .ok out after) :
    out = h.returned ∧ after.payload = ((h.autotunedCode trials c).native abi initial.payload).2 ∧
      after.currentDevice = initial.currentDevice ∧ rt.save h.context = .ok () := by
  unfold run at success
  obtain ⟨u,s1,prepared,rest⟩ := (bind_ok _ _ _ _).mp success
  cases u
  obtain ⟨pvalue,pdevice⟩ := part_success abi _ _ _ _ prepared
  obtain ⟨u,s2,launched,rest⟩ := (bind_ok _ _ _ _).mp rest
  cases u
  have kernel : s2.payload = ((h.autotunedKernelCode trials c).native abi s1.payload).2 ∧
      s2.currentDevice = s1.currentDevice := by
    by_cases launch : r.launch
    · simp only [launch,if_true] at launched
      obtain ⟨inside,body,values,device⟩ := owner_success r.device failures _ s1 s2 launched
      have source := launchPart_success r.device abi _ _ _ _ body
      exact ⟨values.trans source.2.1,device⟩
    · simp only [launch,if_false,Result.ok.injEq,true_and] at launched
      subst s2
      simp [Allocation.autotunedKernelCode,launch,pure,Code.native]
  obtain ⟨u,s3,copied,finish⟩ := (bind_ok _ _ _ _).mp rest
  cases u
  obtain ⟨cvalue,cdevice⟩ := part_success abi _ _ _ _ copied
  obtain ⟨returned,rfl,saved⟩ := finish_success rt h law s3 after out finish
  refine ⟨returned,?_,cdevice.trans (kernel.2.trans pdevice),saved⟩
  rw [cvalue,kernel.1,pvalue]
  simp only [Allocation.autotunedCode,bind,native_bind]

#axiomsClean run_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Checked
