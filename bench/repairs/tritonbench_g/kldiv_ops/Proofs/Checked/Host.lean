import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Finish
namespace VeriTile.Bench.Repairs.KldivOps.Contiguous
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open KldivTriton.Host (World Result ForwardCall BackwardCall Arithmetic Fresh Stable Summed Frame Complement
  ForwardDomain BackwardDomain BackwardReturned bind_ok external_stable)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 8000000

theorem writeView_frame {M α β : Type} (ops : Ops M α β) (out : View)
    (values : Nat → α) (s : World M) (kept : List Allocation) (backed : out.backed)
    (safe : out.numel = 0 ∨ ∀ old ∈ kept, old.separate out.storage) :
    Frame ops kept s {s with memory := writeView ops out values s.memory} := by
  rcases safe with empty | separate
  · simp only [writeView,empty,List.range_zero,stores]
    exact L2NormBwd.Host.Frame.refl _ _ _
  · refine ⟨fun _ h => h,?_,?_,rfl⟩
    · intro old ho key hk
      apply read_frame
      intro i hi
      exact old.spans_separate out.storage (separate old ho) key _ hk (backed i (List.mem_range.mp hi))
    · intro old ho owner address covered
      apply writeView_bytes
      intro i hi
      exact KldivTriton.Host.separate_byte (separate old ho) _ (backed i hi) owner address covered

theorem writeView_complement {M α β : Type} (ops : Ops M α β) (out : View)
    (values : Nat → α) (s : World M) (kept : List Allocation) :
    Complement ops out kept s {s with memory := writeView ops out values s.memory} := by
  refine ⟨fun _ h => h,?_,?_,rfl⟩
  · intro old ho key hk outside
    apply read_frame
    intro i hi
    exact outside i (List.mem_range.mp hi)
  · intro old ho owner address covered outside
    exact writeView_bytes ops out values s.memory owner address outside

theorem forward_admitted (p t : Tensor) (call : ForwardCall) (h : ForwardContract p t Bool.true) :
    checkForward (.tensor p) (.tensor t) (some call.flag)
      (KldivTriton.Contiguous.modeName call.mode) Bool.true = .ok () := by
  apply (python_forward_checks_iff _ _ _ _ _).mpr
  exact ⟨h,rfl,by cases call.mode <;> simp [KldivTriton.Contiguous.validMode,KldivTriton.Contiguous.modeName]⟩

theorem backward_admitted (d t g : Tensor) (call : BackwardCall) (h : BackwardContract d t g) :
    checkBackward (.tensor d) (.tensor t) (.tensor g) (some call.flag) = .ok () :=
  (python_backward_checks_iff _ _ _ _).mpr ⟨h,rfl⟩

def expected {M α β κ : Type} (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ)
    (call : ForwardCall) (eps : α) (s : World M) : Nat → α :=
  forwardExpected ops neg a s.memory call.prediction.view call.target.view call.prediction.cols
    call.flag (reduction call.mode) eps

def expectedBackward {M α β κ : Type} (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ)
    (call : BackwardCall) (s : World M) (i : Nat) : α :=
  derivative a call.flag (logical ops neg s.memory call.target.view i)

/-- Complete checked direct forward call, actual allocation and actual scalar
reduction plan. Entry input values survive all ordinary host effects. -/
theorem forward_success {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (p t : Tensor) (call : ForwardCall) (eps : α) (initial after : World M) (out : View)
    (admit : ForwardContract p t Bool.true) (pb : Binding p call.prediction.view)
    (tb : Binding t call.target.view) (domain : ForwardDomain call initial)
    (success : runForward ops a rt p t call eps initial = .ok out after) :
    ∃ output : View, output.shape = outputShape call ∧
      ForwardReturned ops neg arith call output (expected ops neg a call eps initial) out after ∧
      Frame ops call.caller initial after ∧ after.device = initial.device := by
  have ypLive : initial.live call.prediction.view.storage := domain.live _ (by simp [ForwardCall.caller])
  have tpLive : initial.live call.target.view.storage := domain.live _ (by simp [ForwardCall.caller])
  simp only [runForward,forward_admitted p t call admit] at success
  obtain ⟨output,s1,ez,rest⟩ := (bind_ok _ _ _ _).mp success
  obtain ⟨token,inside,ee,body⟩ := (bind_ok _ _ _ _).mp rest
  obtain ⟨returned,hbody,ex,exitMemory,exitDevice⟩ :=
    KldivTriton.Host.finallyDevice_success ops neg arith rt.base lib.base token _ out after body
  obtain ⟨shape,dense,fresh,allocDevice,_⟩ := lib.zeros _ _ _ _ _ ez
  have fz := lib.zeros_stable (outputShape call) call.prediction.view.storage.owner initial
  simp only [ez,Result.state] at fz
  have fe := external_stable ops neg arith rt.base lib.base (.enter call.prediction.view.storage.owner) (by trivial) s1
  simp only [ee,Result.state] at fe
  obtain ⟨previous,_,_⟩ := lib.base.enter _ _ _ _ ee
  have stable := fz.trans fe
  unfold forwardBody at hbody
  obtain ⟨u,before,setup,finish⟩ := (bind_ok _ _ _ _).mp hbody
  cases u
  have same := lib.setupForward _ _ _ _ setup
  subst before
  let computed : World M := {inside with memory := forwardKernel ops a inside.memory call.prediction.view call.target.view output call.prediction.rows call.prediction.cols call.flag (reduction call.mode) eps}
  have values : ∀ i, i < output.numel → logical ops neg computed.memory output i = expected ops neg a call eps initial i := by
    intro i hi
    have localValue := forward_values ops neg a inside.memory p t call.prediction.view call.target.view output
      call.prediction.rows call.prediction.cols call.flag (reduction call.mode) eps admit.toPairContract pb tb
      call.prediction.shape shape dense fresh.positive i hi
    have transported := forwardExpected_congr ops neg a initial.memory inside.memory call.prediction.view call.target.view output
      call.prediction.rows call.prediction.cols call.flag (reduction call.mode) eps call.prediction.shape
      (tb.shape.trans (admit.shape.trans pb.shape.symm)) shape
      (stable.logical neg call.prediction.view ypLive domain.prediction)
      (stable.logical neg call.target.view tpLive domain.target) i hi
    exact localValue.trans transported
  have finishSpec := finishForward_spec ops neg arith rt lib call output (expected ops neg a call eps initial)
    computed (fe.live _ fresh.live) fresh.backed values
  change finishForward rt call output computed = .ok out returned at finish
  rw [finish] at finishSpec
  obtain ⟨result,ff⟩ := finishSpec
  have fk : Frame ops call.caller inside computed := writeView_frame ops output _ inside call.caller fresh.backed
    (Or.inr (fun old ho => KldivTriton.Host.separate_symm (fresh.separate old (domain.live old ho))))
  have beforeExit := ((stable call.caller domain.live).trans fk).trans
    (ff call.caller (fun old ho => stable.live old (domain.live old ho)))
  have fx := external_stable ops neg arith rt.base lib.base (.exit token) (by trivial) returned
  simp only [ex,Result.state] at fx
  exact ⟨output,shape,result.memory exitMemory (fx.live _ result.live),
    beforeExit.trans (fx call.caller (fun old ho => beforeExit.live old (domain.live old ho))),
    exitDevice.trans (previous.trans allocDevice)⟩

/-- Direct supplied-object writes, the exact scalar-one branch, and the actual
returned cotangent product, with the destination's entire byte complement. -/
theorem backward_success {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (d t g : Tensor) (call : BackwardCall) (initial after : World M) (out : View)
    (admit : BackwardContract d t g) (db : Binding d call.input.view)
    (tb : Binding t call.target.view) (gb : Binding g call.gradient)
    (domain : BackwardDomain call initial)
    (success : runBackward ops a rt d t g call initial = .ok out after) :
    BackwardReturned ops neg arith call.input call.gradient.shape (expectedBackward ops neg a call initial)
      (logical ops neg initial.memory call.gradient) out after ∧
    Complement ops call.input.view call.caller initial after ∧ after.device = initial.device := by
  have inputLive : initial.live call.input.view.storage := domain.live _ (by simp [BackwardCall.caller])
  have targetLive : initial.live call.target.view.storage := domain.live _ (by simp [BackwardCall.caller])
  have gradientLive : initial.live call.gradient.storage := domain.live _ (by simp [BackwardCall.caller])
  have sameSize : call.target.view.numel = call.input.view.numel := by
    simp [View.numel,tb.shape,db.shape,admit.shape]
  simp only [runBackward,backward_admitted d t g call admit] at success
  obtain ⟨token,inside,enter,body⟩ := (bind_ok _ _ _ _).mp success
  obtain ⟨returned,hbody,exit,exitMemory,exitDevice⟩ :=
    KldivTriton.Host.finallyDevice_success ops neg arith rt.base lib.base token _ out after body
  have fe := external_stable ops neg arith rt.base lib.base (.enter call.input.view.storage.owner) (by trivial) initial
  simp only [enter,Result.state] at fe
  obtain ⟨previous,_,_⟩ := lib.base.enter _ _ _ _ enter
  unfold backwardBody at hbody
  obtain ⟨u,before,setup,finish⟩ := (bind_ok _ _ _ _).mp hbody
  cases u
  have same := lib.setupBackward _ _ _ setup
  subst before
  let computed : World M := {inside with memory := backwardKernel ops a inside.memory call.target.view call.input.view call.input.rows call.input.cols call.flag}
  have values : ∀ i, i < call.input.view.numel → logical ops neg computed.memory call.input.view i = expectedBackward ops neg a call initial i := by
    intro i hi
    have v := backward_values ops neg a inside.memory d t call.target.view call.input.view call.input.rows call.input.cols
      call.flag admit.toPairContract db tb call.input.shape i hi
    rw [fe.logical neg call.target.view targetLive domain.target i (by rw [sameSize];exact hi)] at v
    exact v
  have gradientFrame : Frame ops [call.gradient.storage] inside computed := by
    apply writeView_frame ops call.input.view _ inside [call.gradient.storage] domain.input
    by_cases empty : d.metadata.numel = 0
    · exact Or.inl (db.toViewBinding.numel.trans empty)
    · right
      intro old ho
      simp only [List.mem_singleton] at ho
      subst old
      exact KldivTriton.Host.separate_symm (db.toViewBinding.separate gb.toViewBinding (admit.separate empty).2)
  have gvalue : ∀ i, i < call.gradient.numel → logical ops neg computed.memory call.gradient i = logical ops neg initial.memory call.gradient i := by
    intro i hi
    rw [gradientFrame.logical neg call.gradient (by simp) domain.gradient i hi,
      fe.logical neg call.gradient gradientLive domain.gradient i hi]
  change KldivTriton.Host.finishBackward rt.base call call.gradient computed = .ok out returned at finish
  obtain ⟨result,ff⟩ := KldivTriton.Host.finishBackward_success ops neg arith rt.base lib.base call call.gradient
    (expectedBackward ops neg a call initial) (logical ops neg initial.memory call.gradient) computed returned out
    domain.input (fe.live _ inputLive) domain.gradient (fe.live _ gradientLive) values gvalue finish
  have fk : Complement ops call.input.view call.caller inside computed := writeView_complement ops call.input.view _ inside call.caller
  have beforeExit := ((Complement.ofFrame call.input.view (fe call.caller domain.live)).trans fk).trans
    (.ofFrame _ (ff call.caller (fun old ho => fe.live old (domain.live old ho))))
  have fx := external_stable ops neg arith rt.base lib.base (.exit token) (by trivial) returned
  simp only [exit,Result.state] at fx
  exact ⟨result.memory exitMemory (fx.live _ result.live),
    beforeExit.trans (.ofFrame _ (fx call.caller (fun old ho => beforeExit.live old (domain.live old ho)))),
    exitDevice.trans previous⟩

#axiomsClean forward_success
#axiomsClean backward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous
