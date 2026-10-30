import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Checked.Binding

namespace VeriTile.Bench.Repairs.KldivTriton.Contiguous
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Host
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 8000000

def modeName : Math.Mode → String
  | .none => "none" | .sum => "sum" | .mean => "mean" | .batchmean => "batchmean"

def forwardBody {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (rt : Runtime M) (call : ForwardCall)
    (output : Matrix) (s : World M) : Result M View :=
  let p : Physical.ForwardLayout := ⟨call.prediction,call.target,output,
    blockSize call.prediction.cols,call.flag,modeCode call.mode⟩
  (rt.perform (.setupForward p s.queue) s).bind fun _ before =>
    finishForward rt call output {before with memory := Storage.forwardKernel ops adapter before.memory p (block_positive _)}

def runForward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (rt : Runtime M) (a b : Tensor)
    (call : ForwardCall) (s : World M) : Result M View :=
  match checkForward (.tensor a) (.tensor b) (some call.flag) (modeName call.mode) with
  | .error reason => .error reason s
  | .ok () =>
    (rt.perform (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) s).bind fun output allocated =>
      (rt.perform (.enter call.prediction.view.storage.owner) allocated).bind fun token inside =>
        finallyDevice rt token (forwardBody ops adapter rt call output inside)

def backwardBody {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (rt : Runtime M) (call : BackwardCall)
    (s : World M) : Result M View :=
  let p : Physical.BackwardLayout := ⟨call.target,call.input,blockSize call.input.cols,call.flag⟩
  (rt.perform (.setupBackward p s.queue) s).bind fun _ before =>
    finishBackward rt call call.gradient {before with memory := Storage.backwardKernel ops adapter before.memory p (block_positive _)}

def runBackward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (rt : Runtime M) (a b g : Tensor)
    (call : BackwardCall) (s : World M) : Result M View :=
  match checkBackward (.tensor a) (.tensor b) (.tensor g) (some call.flag) with
  | .error reason => .error reason s
  | .ok () =>
    (rt.perform (.enter call.input.view.storage.owner) s).bind fun token inside =>
      finallyDevice rt token (backwardBody ops adapter rt call inside)

theorem forward_admitted (a b : Tensor) (call : ForwardCall) (h : ForwardContract a b) :
    checkForward (.tensor a) (.tensor b) (some call.flag) (modeName call.mode) = .ok () := by
  apply (python_forward_checks_iff _ _ _ _).mpr
  exact ⟨h,rfl,by cases call.mode <;> simp [validMode,modeName]⟩

theorem backward_admitted (a b g : Tensor) (call : BackwardCall) (h : BackwardContract a b g) :
    checkBackward (.tensor a) (.tensor b) (.tensor g) (some call.flag) = .ok () :=
  (python_backward_checks_iff _ _ _ _).mpr ⟨h,rfl⟩

/-- Direct inputs, the actual fresh output, and no materialization effects. -/
theorem forward_success {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Storage.Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (a b : Tensor) (call : ForwardCall) (initial after : World M) (out : View)
    (admit : ForwardContract a b) (ya : MatrixBinding a call.prediction)
    (ta : MatrixBinding b call.target) (domain : ForwardDomain call initial)
    (success : runForward ops adapter rt a b call initial = .ok out after) :
    ∃ output : Matrix, output.rows = call.prediction.rows ∧ output.cols = outputCols call ∧
      ForwardReturned ops neg arith call output (forwardExpected ops neg adapter call initial) out after ∧
      Frame ops call.caller initial after ∧ after.device = initial.device := by
  have ypLive : initial.live call.prediction.view.storage := domain.live _ (by simp [ForwardCall.caller])
  have tpLive : initial.live call.target.view.storage := domain.live _ (by simp [ForwardCall.caller])
  obtain ⟨rows,cols⟩ := ya.rows_cols ta admit.shape
  simp only [runForward,forward_admitted a b call admit] at success
  obtain ⟨output,s1,ez,rest⟩ := (bind_ok _ _ _ _).mp success
  obtain ⟨token,inside,ee,body⟩ := (bind_ok _ _ _ _).mp rest
  obtain ⟨returned,hbody,ex,exitMemory,exitDevice⟩ :=
    finallyDevice_success ops neg arith rt lib token _ out after body
  obtain ⟨outRows,outCols,_,unique,fresh,_⟩ := lib.zeros _ _ _ _ _ _ ez
  have fz := external_stable ops neg arith rt lib
    (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) (by trivial) initial
  simp only [ez,Result.state] at fz
  have fe := external_stable ops neg arith rt lib (.enter call.prediction.view.storage.owner) (by trivial) s1
  simp only [ee,Result.state] at fe
  obtain ⟨previous,_,_⟩ := lib.enter _ _ _ _ ee
  have beforeStable := fz.trans fe
  unfold forwardBody at hbody
  obtain ⟨u,before,setup,finish⟩ := (bind_ok _ _ _ _).mp hbody
  cases u
  have same := lib.setupForward _ _ _ _ setup
  subst before
  have yValues := (ya.prepared ops neg admit.left initial ypLive).transport beforeStable
  have tValues := (ta.prepared ops neg admit.right initial tpLive).transport beforeStable
  let computed : World M := {inside with memory := Storage.forwardKernel ops adapter inside.memory ⟨call.prediction,call.target,output,blockSize call.prediction.cols,call.flag,modeCode call.mode⟩ (block_positive _)}
  have values : ∀ i, i < output.view.numel → logical ops neg computed.memory output.view i =
      forwardExpected ops neg adapter call initial i := by
    intro i hi
    exact forward_kernel_values ops neg adapter call initial inside call.prediction call.target output
      yValues tValues rows cols outRows outCols fresh.positive unique i hi
  have outputLive : computed.live output.view.storage := fe.live _ fresh.live
  have finishSpec := finishForward_spec ops neg arith rt lib call output
    (forwardExpected ops neg adapter call initial) computed outputLive fresh.backed values
  change finishForward rt call output computed = .ok out returned at finish
  rw [finish] at finishSpec
  obtain ⟨result,ff⟩ := finishSpec
  have fk : Frame ops call.caller inside computed := forward_frame ops adapter _ (block_positive _) inside call.caller
    (output.from_backed fresh.backed) (fun old ho => separate_symm (fresh.separate old (domain.live old ho)))
  have callerBefore : Frame ops call.caller initial returned :=
    ((beforeStable call.caller domain.live).trans fk).trans
      (ff call.caller (fun old ho => beforeStable.live old (domain.live old ho)))
  have fx := external_stable ops neg arith rt lib (.exit token) (by trivial) returned
  simp only [ex,Result.state] at fx
  refine ⟨output,outRows,outCols,result.memory exitMemory (fx.live _ result.live),
    callerBefore.trans (fx call.caller (fun old ho => callerBefore.live old (domain.live old ho))),?_⟩
  have d0 := lib.device (.zeros call.prediction.rows (outputCols call) call.prediction.view.storage.owner) initial
  simp only [ez,Result.state] at d0
  exact exitDevice.trans (previous.trans d0)

/-- Empty direct destinations write no cells and need no separation premise. -/
theorem backward_frame_direct {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (adapter : Storage.Adapter α κ) (p : Physical.BackwardLayout)
    (positive : 0 < p.block) (before : World M) (kept : List Allocation)
    (backed : p.work.backed)
    (safe : p.work.view.numel = 0 ∨ ∀ old ∈ kept, old.separate p.work.view.storage) :
    Frame ops kept before {before with memory := Storage.backwardKernel ops adapter before.memory p positive} := by
  rcases safe with empty | separate
  · rw [p.work.numel] at empty
    have jobs : Physical.jobs p.work.rows p.work.cols = [] := by
      rcases Nat.mul_eq_zero.mp empty with h | h <;> simp [Physical.jobs,List.product,h]
    simp only [Storage.backwardKernel,jobs,stores]
    exact L2NormBwd.Host.Frame.refl ops kept before.toWorld
  · exact backward_frame ops adapter p positive before kept backed separate

/-- The complete byte complement of the directly written destination. -/
theorem backward_kernel_complement {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (adapter : Storage.Adapter α κ) (p : Physical.BackwardLayout)
    (positive : 0 < p.block) (before : World M) (kept : List Allocation) :
    Complement ops p.work.view kept before
      {before with memory := Storage.backwardKernel ops adapter before.memory p positive} := by
  refine ⟨fun _ h => h,?_,?_,rfl⟩
  · intro old ho key hk outside
    apply read_frame
    intro rc hrc
    obtain ⟨hr,hc⟩ := (Physical.job_member _ _ _ _).mp hrc
    have h := outside (rc.1*p.work.cols+rc.2) (coord_bound p.work _ _ hr hc)
    simpa only [p.work.logical_key _ _ hc] using h
  · intro old ho owner address covered outside
    apply Storage.backward_bytes
    intro r c hr hc
    have h := outside (r*p.work.cols+c) (coord_bound p.work _ _ hr hc)
    simpa only [p.work.logical_key _ _ hc] using h

/-- Direct backward writes preserve the entry target/cotangent and return the
supplied object for scalar-one, otherwise the actual Torch product. -/
theorem backward_success {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Storage.Adapter α κ)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (a b g : Tensor) (call : BackwardCall) (initial after : World M) (out : View)
    (admit : BackwardContract a b g) (ia : MatrixBinding a call.input)
    (ta : MatrixBinding b call.target) (ga : ViewBinding g call.gradient)
    (domain : BackwardDomain call initial)
    (success : runBackward ops adapter rt a b g call initial = .ok out after) :
    BackwardReturned ops neg arith call.input call.gradient.shape
      (backwardExpected ops neg adapter call initial)
      (fun i => logical ops neg initial.memory call.gradient i) out after ∧
    Complement ops call.input.view call.caller initial after ∧ after.device = initial.device := by
  have inputLive : initial.live call.input.view.storage := domain.live _ (by simp [BackwardCall.caller])
  have targetLive : initial.live call.target.view.storage := domain.live _ (by simp [BackwardCall.caller])
  have gradientLive : initial.live call.gradient.storage := domain.live _ (by simp [BackwardCall.caller])
  obtain ⟨rows,cols⟩ := ia.rows_cols ta admit.shape
  simp only [runBackward,backward_admitted a b g call admit] at success
  obtain ⟨token,inside,enter,body⟩ := (bind_ok _ _ _ _).mp success
  obtain ⟨returned,hbody,exit,exitMemory,exitDevice⟩ := finallyDevice_success ops neg arith rt lib token _ out after body
  have fe := external_stable ops neg arith rt lib (.enter call.input.view.storage.owner) (by trivial) initial
  simp only [enter,Result.state] at fe
  obtain ⟨previous,_,_⟩ := lib.enter _ _ _ _ enter
  unfold backwardBody at hbody
  obtain ⟨u,before,setup,finish⟩ := (bind_ok _ _ _ _).mp hbody
  cases u
  have same := lib.setupBackward _ _ _ _ setup
  subst before
  let p : Physical.BackwardLayout := ⟨call.target,call.input,blockSize call.input.cols,call.flag⟩
  let computed : World M := {inside with memory := Storage.backwardKernel ops adapter inside.memory p (block_positive _)}
  have targetValues := (ta.prepared ops neg admit.right initial targetLive).transport fe
  have values : ∀ i, i < call.input.view.numel → logical ops neg computed.memory call.input.view i =
      backwardExpected ops neg adapter call initial i := by
    intro i hi
    exact backward_kernel_values ops neg adapter call initial inside call.target call.input targetValues
      rows.symm cols.symm (ia.negative.trans admit.left.noNeg) (ia.contiguous admit.left.contiguous).2 i hi
  have gradientFrame : Frame ops [call.gradient.storage] inside computed := by
    apply backward_frame_direct ops adapter p (block_positive _) inside [call.gradient.storage] (call.input.from_backed domain.input)
    by_cases empty : a.metadata.numel = 0
    · exact Or.inl (ia.toViewBinding.numel.trans empty)
    · right
      intro old ho
      simp only [List.mem_singleton] at ho
      subst old
      exact separate_symm (ia.toViewBinding.separate ga (admit.separate empty).2)
  have gvalue : ∀ i, i < call.gradient.numel → logical ops neg computed.memory call.gradient i =
      logical ops neg initial.memory call.gradient i := by
    intro i hi
    rw [gradientFrame.logical neg call.gradient (by simp) domain.gradient i hi,
      fe.logical neg call.gradient gradientLive domain.gradient i hi]
  change finishBackward rt call call.gradient computed = .ok out returned at finish
  obtain ⟨result,ff⟩ := finishBackward_success ops neg arith rt lib call call.gradient
    (backwardExpected ops neg adapter call initial) (fun i => logical ops neg initial.memory call.gradient i)
    computed returned out domain.input (fe.live _ inputLive) domain.gradient (fe.live _ gradientLive) values gvalue finish
  have fk : Complement ops call.input.view call.caller inside computed :=
    backward_kernel_complement ops adapter p (block_positive _) inside call.caller
  have callerBefore := ((Complement.ofFrame call.input.view (fe call.caller domain.live)).trans fk).trans
    (.ofFrame _ (ff call.caller (fun old ho => fe.live old (domain.live old ho))))
  have fx := external_stable ops neg arith rt lib (.exit token) (by trivial) returned
  simp only [exit,Result.state] at fx
  exact ⟨result.memory exitMemory (fx.live _ result.live),
    callerBefore.trans (.ofFrame _ (fx call.caller (fun old ho => callerBefore.live old (domain.live old ho)))),
    exitDevice.trans previous⟩

#axiomsClean forward_success
#axiomsClean backward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Contiguous
