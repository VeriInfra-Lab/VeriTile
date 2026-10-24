import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Autotune

/- The source launcher selects a.device through the standard CUDA device
context. Formal device/stream observations refer to that effective owning
device; leaving the context restores the ambient caller device. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
open VeriTile.Triton FP.Structural Whole
noncomputable section

/-- One actual autotuned call may report either compiler or runtime failure.
The successful compiler/runtime implementation is the shared typed-execution
boundary; no native numerical correctness is assumed by these error tags. -/
inductive BackendError where
  | compilation (message : String)
  | launch (message : String)
  deriving DecidableEq

inductive WrapperError where
  | incompatibleDimensions | noncontiguousA
  | resolveA (message : String) | resolveB (message : String)
  | allocation (message : String) | backend (error : BackendError)
  | executionFailure
  deriving DecidableEq

/-- Ordinary API operations for this invocation. Resolve calls occur in Python
tuple-expression order. Allocation returns its actual object identity. The
autotuned call is reached only after the source empty-output check. On success
its result records the finite benchmark trial configuration list; r.config is
the selected final configuration. The list may be empty or contain repetitions.
The numerical grids themselves are executed by runTrials and host below. -/
structure WrapperAPI (α : Type) where
  resolveA : State α → Except String (State α)
  resolveB : State α → Except String (State α)
  allocateC : State α → Except String (RegionName × State α)
  compileLaunch : State α → Except BackendError (List Config)

def prepare {α : Type} (api : WrapperAPI α) (r : Request) (before : State α) :
    Except WrapperError (RegionName × State α) :=
  if r.inner≠r.innerB then .error .incompatibleDimensions
  else if r.aContiguous=Bool.false then .error .noncontiguousA
  else match api.resolveA before with
    | .error e => .error (.resolveA e)
    | .ok a => match api.resolveB a with
      | .error e => .error (.resolveB e)
      | .ok b => match api.allocateC b with
        | .error e => .error (.allocation e)
        | .ok output => .ok output

def allocatedRequest (r : Request) (c : RegionName) : Request := {r with C := c}

@[simp] theorem allocatedRequest_self (r : Request) : allocatedRequest r r.C=r := by
  cases r
  rfl

/-- Source-ordered wrapper. Its success path runs the proved actual candidate
typed kernel using the allocator's returned output. API errors remain distinct
outcomes; no successful preparation/launch is a theorem premise. -/
def wrapper {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (api : WrapperAPI α) (r : Request) (before : State α) :
    Except WrapperError (RegionName × State α) :=
  match prepare api r before with
  | .error e => .error e
  | .ok (c, prepared) =>
    if r.rows=0 ∨ r.columns=0 then .ok (c,prepared)
    else match api.compileLaunch prepared with
      | .error e => .error (.backend e)
      | .ok trials => match runTrials M D (allocatedRequest r c) trials prepared with
        | none => .error .executionFailure
        | some after => match host M D (allocatedRequest r c) after with
          | .error _ => .error .executionFailure
          | .ok output => .ok output

structure PreparedBy {α : Type} (api : WrapperAPI α) (r : Request)
    (before : State α) (c : RegionName) (prepared : State α) : Prop where
  dimensions : r.inner=r.innerB
  contiguous : r.aContiguous=Bool.true
  stages : ∃ a b, api.resolveA before=.ok a ∧ api.resolveB a=.ok b ∧
    api.allocateC b=.ok (c,prepared)

/-- Standard successful resolve_neg / fresh torch.empty contract. It covers
only library preparation, not kernel success, output values or a kernel frame.
Fresh output initialization is allowed by ResolveNegContract.frame.
The tuner supplies legal source configurations and the installed CUDA/HIP
launcher's signed-int grid ABI for every trial and the chosen final launch;
oversized valid tensors remain in the error domain, rather than being removed
by an input precondition. -/
structure WrapperAPIContract {α : Type} [Inhabited α] (api : WrapperAPI α)
    (r : Request) (before : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α) : Prop where
  preparation : ∀ c prepared, PreparedBy api r before c prepared →
    c=r.C ∧ ∃ tensor : TensorAPI r prepared,
      Nonempty (ResolveNegContract r before prepared tensor A B)
  trialsLegal : ∀ prepared trials, api.compileLaunch prepared=.ok trials →
    ∀ config∈trials, config∈configurations r.target
  trialABI : ∀ prepared trials, api.compileLaunch prepared=.ok trials →
    ∀ config∈trials, programs (withConfig r config)<2^31
  launchABI : ∀ prepared trials, api.compileLaunch prepared=.ok trials → programs r<2^31

inductive PreparationFailure {α : Type} (api : WrapperAPI α) (r : Request)
    (before : State α) : WrapperError → Prop
  | dimensions (h : r.inner≠r.innerB) : PreparationFailure api r before .incompatibleDimensions
  | contiguous (hd : r.inner=r.innerB) (hc : r.aContiguous=Bool.false) :
      PreparationFailure api r before .noncontiguousA
  | resolveA (hd : r.inner=r.innerB) (hc : r.aContiguous=Bool.true)
      (e : String) (he : api.resolveA before=.error e) :
      PreparationFailure api r before (.resolveA e)
  | resolveB (hd : r.inner=r.innerB) (hc : r.aContiguous=Bool.true)
      (a : State α) (ha : api.resolveA before=.ok a)
      (e : String) (he : api.resolveB a=.error e) :
      PreparationFailure api r before (.resolveB e)
  | allocation (hd : r.inner=r.innerB) (hc : r.aContiguous=Bool.true)
      (a b : State α) (ha : api.resolveA before=.ok a) (hb : api.resolveB a=.ok b)
      (e : String) (he : api.allocateC b=.error e) :
      PreparationFailure api r before (.allocation e)

inductive WrapperFailure {α : Type} (api : WrapperAPI α) (r : Request)
    (before : State α) : WrapperError → Prop
  | preparation {e : WrapperError} (h : PreparationFailure api r before e) :
      WrapperFailure api r before e
  | backend (c : RegionName) (prepared : State α)
      (done : PreparedBy api r before c prepared)
      (nonempty : ¬(r.rows=0 ∨ r.columns=0))
      (e : BackendError) (failed : api.compileLaunch prepared=.error e) :
      WrapperFailure api r before (.backend e)

theorem prepare_cases {α : Type} (api : WrapperAPI α) (r : Request) (before : State α) :
    (∃ e, prepare api r before=.error e ∧ PreparationFailure api r before e) ∨
    (∃ c prepared, prepare api r before=.ok (c,prepared) ∧ PreparedBy api r before c prepared) := by
  by_cases hd : r.inner=r.innerB
  · by_cases hc : r.aContiguous=Bool.false
    · exact Or.inl ⟨_,by simp [prepare,hd,hc],.contiguous hd hc⟩
    · have ht : r.aContiguous=Bool.true := by cases h : r.aContiguous <;> simp_all
      cases ha : api.resolveA before with
      | error e => exact Or.inl ⟨_,by simp [prepare,hd,ht,ha],.resolveA hd ht e ha⟩
      | ok a =>
        cases hb : api.resolveB a with
        | error e => exact Or.inl ⟨_,by simp [prepare,hd,ht,ha,hb],.resolveB hd ht a ha e hb⟩
        | ok b =>
          cases hp : api.allocateC b with
          | error e => exact Or.inl ⟨_,by simp [prepare,hd,ht,ha,hb,hp],.allocation hd ht a b ha hb e hp⟩
          | ok output =>
            rcases output with ⟨c,p⟩
            exact Or.inr ⟨c,p,by simp [prepare,hd,ht,ha,hb,hp],⟨hd,ht,a,b,ha,hb,hp⟩⟩
  · exact Or.inl ⟨_,by simp [prepare,hd],.dimensions hd⟩

/-- The full successful postcondition includes the concrete preparation trace,
every logical output, both original allocation frames, the complete pre-host
frame, pointer/control safety, the native launch-ID bound when launched, and
the entire trial sequence followed by the chosen final grid. -/
def WrapperSuccess {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α)
    (api : WrapperAPI α) (r : Request) (before : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (c : RegionName) (final : State α) : Prop :=
  c=r.C ∧ ∃ (prepared : State α) (tensor : TensorAPI r prepared)
    (resolve : ResolveNegContract r before prepared tensor A B),
    PreparedBy api r before c prepared ∧
    (∀ i : Fin r.rows, ∀ j : Fin r.columns,
      final.mem r.C (outputOffset r i.val j.val)=Cell.mk .fp16 (logicalReference Q r A B i j)) ∧
    (∀ region offset, ¬Footprint r region offset → final.mem region offset=prepared.mem region offset) ∧
    (∀ region offset, ¬Footprint r region offset →
      ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
      final.mem region offset=before.mem region offset) ∧
    (∀ offset, final.mem resolve.originalA offset=before.mem resolve.originalA offset) ∧
    (∀ offset, final.mem resolve.originalB offset=before.mem resolve.originalB offset) ∧
    AddressSafety r prepared tensor ∧
    (¬(r.rows=0 ∨ r.columns=0) → programs r<2^31) ∧
    Native.FullAccessSafety r prepared tensor ∧
    (∃ trials, (r.rows=0 ∨ r.columns=0 → trials=[]) ∧
      (¬(r.rows=0 ∨ r.columns=0) → api.compileLaunch prepared=.ok trials) ∧
      TuningExecution M D r prepared tensor trials final)

def WrapperOutcome {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α)
    (api : WrapperAPI α) (r : Request) (before : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α) : Prop :=
  match wrapper M D api r before with
  | .error e => WrapperFailure api r before e
  | .ok (c,final) => WrapperSuccess M D Q api r before A B c final

/-- Total source-ordered result. Preparation and launch can fail; only their
ordinary API contracts are assumed. All typed execution, output and kernel
frame facts are supplied by proved trials_frame and resolved_fp. The latter
uses the preserved input arrays and a fresh accumulator, so trial writes to C
cannot influence the final selected configuration's output. -/
theorem wrapper_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (api : WrapperAPI α) (r : Request) (before : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (apiLaw : WrapperAPIContract api r before A B) :
    WrapperOutcome M D Q api r before A B := by
  rcases prepare_cases api r before with ⟨e,he,hfailure⟩ | ⟨c,p,hp,hprepared⟩
  · simp only [WrapperOutcome,wrapper,he]
    exact .preparation hfailure
  · obtain ⟨hc,t,⟨resolve⟩⟩ := apiLaw.preparation c p hprepared
    subst c
    by_cases empty : r.rows=0 ∨ r.columns=0
    · obtain ⟨final,he,hvalue,hframe,hcaller,hsafety⟩ := resolved_fp M D Q law r before p t A B resolve
      obtain ⟨hA,hB⟩ := original_input_frame r before p final t A B resolve hcaller
      have hid := empty_output_identity M D r p hprepared.dimensions hprepared.contiguous empty
      have hfinal : final=p := by
        have hh := he.symm.trans hid
        exact congrArg Prod.snd (Except.ok.inj hh)
      subst final
      have hgrid : programs r<2^31 := by rw [Native.empty_grid r empty]; decide
      have htrace := tuning_execution M D r p p p t [] (by simp) (by simp)
        hgrid rfl (fun _ _ _ => rfl) he
      simp only [WrapperOutcome,wrapper,hp,if_pos empty]
      exact ⟨rfl,p,t,resolve,hprepared,hvalue,hframe,hcaller,hA,hB,hsafety,
        (fun nonempty => False.elim (nonempty empty)),Native.full_access_safety r p t hgrid,
        [],(fun _ => rfl),(fun nonempty => False.elim (nonempty empty)),htrace⟩
    · cases hl : api.compileLaunch p with
      | error e =>
        simp only [WrapperOutcome,wrapper,hp,if_neg empty,hl]
        exact .backend r.C p hprepared empty e hl
      | ok trials =>
        obtain ⟨after,htrials,htrialframe⟩ := trials_frame M D r trials p
        let ta := t.afterTrials htrialframe
        let ra := resolve.afterTrials htrialframe
        obtain ⟨final,he,hvalue,hframe,hcaller,_⟩ := resolved_fp M D Q law r before after ta A B ra
        obtain ⟨hA,hB⟩ := original_input_frame r before p final t A B resolve hcaller
        have hgrid := apiLaw.launchABI p trials hl
        have htrace := tuning_execution M D r p after final t trials
          (apiLaw.trialsLegal p trials hl) (apiLaw.trialABI p trials hl)
          hgrid htrials htrialframe he
        simp only [WrapperOutcome,wrapper,hp,if_neg empty,hl,allocatedRequest_self,htrials,he]
        exact ⟨rfl,p,t,resolve,hprepared,hvalue,
          (fun region offset hout => (hframe region offset hout).trans (htrialframe region offset hout)),
          hcaller,hA,hB,address_safety r p t,
          (fun _ => hgrid),Native.full_access_safety r p t hgrid,
          trials,(fun h => False.elim (empty h)),(fun _ => hl),htrace⟩

/-- The wrapper never reports a modeled executionFailure, independently of all
six numerical ports. Any error is the first reached assertion/API failure. -/
theorem wrapper_executes {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α)
    (api : WrapperAPI α) (r : Request) (before : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (apiLaw : WrapperAPIContract api r before A B) :
    match wrapper M D api r before with
    | .error e => WrapperFailure api r before e
    | .ok (c,_) => c=r.C := by
  rcases prepare_cases api r before with ⟨e,he,hfailure⟩ | ⟨c,p,hp,hprepared⟩
  · simp only [wrapper,he]
    exact .preparation hfailure
  · obtain ⟨hc,_⟩ := apiLaw.preparation c p hprepared
    subst c
    by_cases empty : r.rows=0 ∨ r.columns=0
    · simp [wrapper,hp,empty]
    · cases hl : api.compileLaunch p with
      | error e =>
        simp only [wrapper,hp,if_neg empty,hl]
        exact .backend r.C p hprepared empty e hl
      | ok trials =>
        obtain ⟨after,htrials,_⟩ := trials_frame M D r trials p
        obtain ⟨final,he⟩ := host_executes M D r after hprepared.dimensions hprepared.contiguous
        simp [wrapper,hp,empty,hl,htrials,he]

theorem wrapper_dimension_error {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (api : WrapperAPI α)
    (r : Request) (before : State α) (h : r.inner≠r.innerB) :
    wrapper M D api r before=.error .incompatibleDimensions := by simp [wrapper,prepare,h]

theorem wrapper_contiguity_error {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (api : WrapperAPI α)
    (r : Request) (before : State α) (hd : r.inner=r.innerB) (hc : r.aContiguous=Bool.false) :
    wrapper M D api r before=.error .noncontiguousA := by simp [wrapper,prepare,hd,hc]

theorem wrapper_empty_return {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (api : WrapperAPI α)
    (r : Request) (before prepared : State α) (c : RegionName)
    (hp : prepare api r before=.ok (c,prepared)) (empty : r.rows=0 ∨ r.columns=0) :
    wrapper M D api r before=.ok (c,prepared) := by simp [wrapper,hp,empty]

theorem successful_pid_machine {α : Type} [Inhabited α]
    (api : WrapperAPI α) (r : Request) (before prepared : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (apiLaw : WrapperAPIContract api r before A B)
    (trials : List Config) (launched : api.compileLaunch prepared=.ok trials)
    (pid : Nat) (hp : pid<programs r) :
    pid<2^31 := lt_trans hp (apiLaw.launchABI prepared trials launched)

#axiomsClean wrapper_correct
#axiomsClean wrapper_executes
#axiomsClean prepare_cases
#axiomsClean successful_pid_machine
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
