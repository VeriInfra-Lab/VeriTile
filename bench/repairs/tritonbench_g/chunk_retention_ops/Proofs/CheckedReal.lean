import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Host
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Preparation
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.InitialReturned
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.RealReference
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.LossPolynomial
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Checked

def realForwardEntry (c : Shape) (storage : Storage c) (scale : ℝ)
    (compiler : Compiler c tree scale seed rf terminal ri) (before : Memory ℝ) :=
  (forwardTensors storage rf,execute realOps c storage.binding scale compiler.forward before)
def realBackwardEntry (c : Shape) (storage : Storage c) (scale : ℝ)
    (compiler : Compiler c tree scale seed rf terminal ri) (checkpoint : Bool) (later : Memory ℝ) :=
  (backwardTensors storage ri,execute realOps c storage.binding scale
    (recomputeCode c tree checkpoint++compiler.backward) later)

theorem real_forward_output (c : Shape) (storage : Storage c) (scale : ℝ)
    (compiler : Compiler c tree scale seed rf terminal ri) (before : Memory ℝ) :
    ∀ h t j,h<c.heads → t<c.T → j<c.V →
      (realForwardEntry c storage scale compiler before).2
        (storage.base .output+c.valueOffset h t j)=
      MathTarget.output c (inputsOfHeap (storage.binding.view 0 before)) scale Bool.true h t j := by
  intro h t j hh ht hj
  simp only [realForwardEntry,compiler.forward_eq,forward_graph]
  have value := output_value realOps c storage.binding scale tree rf before h t j hh ht hj
  rw [RealScalar.reference_output c _ scale h t j ht hj] at value
  simpa only [Storage.binding] using value

theorem real_forward_final (c : Shape) (storage : Storage c) (scale : ℝ)
    (compiler : Compiler c tree scale seed Bool.true terminal ri) (before : Memory ℝ) :
    ∀ h e j,h<c.heads → e<c.K → j<c.V →
      (realForwardEntry c storage scale compiler before).2
        (storage.base .finalState+c.stateOffset h e j)=
      MathTarget.finalState c (inputsOfHeap (storage.binding.view 0 before)) Bool.true h e j := by
  intro h e j hh he hj
  simp only [realForwardEntry,compiler.forward_eq,forward_graph]
  have value := final_value realOps c storage.binding scale tree before h e j hh he hj
  rw [RealScalar.reference_final c _ h e j he hj] at value
  simpa only [Storage.binding] using value

def realGradientTarget (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (a : Address) : ℝ :=
  if a.region=.dq then MathTarget.queryGradient c xs scale Bool.true a.head a.row a.col
  else if a.region=.dk then MathTarget.keyGradient c xs scale a.head a.row a.col
  else MathTarget.valueGradient c xs scale a.head a.row a.col

theorem real_backward_gradients (c : Shape) (fstorage storage : Storage c) (scale : ℝ)
    (compiler : Compiler c tree scale seed rf terminal ri) (checkpoint : Bool) (before later : Memory ℝ)
    (versions : Versions c (fstorage.binding.view 0 (realForwardEntry c fstorage scale compiler before).2)
      (storage.binding.view 0 later) checkpoint)
    (a : Address) (valid : a.Valid c) (tag : a.region∈[Region.dq,Region.dk,Region.dv]) :
    (realBackwardEntry c storage scale compiler checkpoint later).2 (storage.binding.address a)=
      realGradientTarget c (Separate.joined (fstorage.binding.view 0 before) (storage.binding.view 0 later)) scale a := by
  have completed : fstorage.binding.view 0 (realForwardEntry c fstorage scale compiler before).2=
      Source.afterO realOps c scale tree rf (fstorage.binding.view 0 before) := by
    simp only [realForwardEntry,compiler.forward_eq,forward_graph]
    exact Separate.forward_physical _ _ _ _ _ _ _
  have post := backward_saved realOps c scale tree compiler.sumCoverage rf ri checkpoint
    fstorage storage before _ later completed versions
  change Separate.BackwardPost realOps c _ scale tree
    (storage.binding.view 0 (execute realOps c storage.binding scale _ later)) at post
  have readonly : a.region∉inputRegions := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at tag
    rcases tag with h|h|h <;> simp [h,inputRegions]
  have read := storage.binding.view_read (0:ℝ)
    (realBackwardEntry c storage scale compiler checkpoint later).2 a valid readonly
  rw [←read]
  simp only [realBackwardEntry,compiler.backward_eq]
  rcases a with ⟨region,h,s,t,j⟩
  simp only [List.mem_cons,List.not_mem_nil,or_false] at tag
  rcases tag with rfl|rfl|rfl <;> simp only [Address.Valid] at valid
  all_goals rcases valid with ⟨hh,rfl,ht,hj⟩
  · rw [post.query h hh t ht j hj,RealScalar.reference_query c _ scale h t j ht hj]
    rfl
  · rw [post.key h hh t ht j hj,RealScalar.reference_key c _ scale h t j ht hj]
    rfl
  · rw [post.value h hh t ht j hj,RealScalar.reference_value c _ scale tree compiler.sumCoverage h t j ht hj]
    rfl

theorem real_backward_initial (c : Shape) (storage : Storage c) (scale : ℝ)
    (compiler : Compiler c tree scale seed rf terminal Bool.true) (checkpoint : Bool) (later : Memory ℝ) :
    ∀ h e j,h<c.heads → e<c.K → j<c.V →
      (realBackwardEntry c storage scale compiler checkpoint later).2
        (storage.base .initialGradient+c.stateOffset h e j)=
      MathTarget.initialGradient c (inputsOfHeap (storage.binding.view 0 later)) scale h e j := by
  intro h e j hh he hj
  simp only [realBackwardEntry,compiler.backward_eq]
  rw [initial_value realOps c storage scale tree checkpoint later h e j hh he hj,
    RealScalar.real_cast,RealScalar.reverse_scalar c _ scale h c.chunks e j le_rfl he hj,
    ScalarSource.initial_gradient_correct]
  rfl

theorem real_forward_optional_final (c : Shape) (storage : Storage c) (scale : ℝ)
    (compiler : Compiler c tree scale seed rf terminal ri) (before : Memory ℝ)
    (enabled : rf=Bool.true) :
    ∀ h e j,h<c.heads → e<c.K → j<c.V →
      (realForwardEntry c storage scale compiler before).2
        (storage.base .finalState+c.stateOffset h e j)=
      MathTarget.finalState c (inputsOfHeap (storage.binding.view 0 before)) Bool.true h e j := by
  cases rf <;> simp only [Bool.false_eq_true] at enabled
  exact real_forward_final c storage scale compiler before

/-- Complete checked forward API in exact Real arithmetic, preserving the
actual returned Tensor objects. The target is the independent dense retention
formula over original logical reads, not the scalar source recurrence. -/
specification chunk_retention_ops_repaired_real_returns
    (a : Contract.Forward) (admitted : Contract.ForwardContract a) (call : ForwardCall a admitted)
    (compiler : Compiler (shape a admitted) tree a.scale.value a.initial.isSome a.final terminal ri)
    (original before : Memory ℝ)
    (preparation : PreparedInputs (shape a admitted) call.storage (0:ℝ)
      original before (forwardObjects a)) :
    (Contract.forwardCheck a=Bool.true ↔ Contract.ForwardContract a) ∧
    (realForwardEntry _ call.storage a.scale.value compiler before).1=forwardTensors call.storage a.final ∧
    ((realForwardEntry _ call.storage a.scale.value compiler before).1.1).metadata.shape=Contract.valueShape a.dims ∧
    ((realForwardEntry _ call.storage a.scale.value compiler before).1.1).metadata.dtype=.floating .fp32 ∧
    ((realForwardEntry _ call.storage a.scale.value compiler before).1.1).metadata.device=a.k.metadata.device ∧
    (∀ h t j,h<(shape a admitted).heads → t<a.dims.T → j<a.dims.V →
      (realForwardEntry _ call.storage a.scale.value compiler before).2
        (call.storage.base .output+(shape a admitted).valueOffset h t j)=
      MathTarget.output (shape a admitted)
        (inputsOfHeap (originalHeap (shape a admitted) (0:ℝ) original (forwardObjects a)))
        a.scale.value Bool.true h t j) ∧
    (a.final=Bool.true → ∀ h e j,h<(shape a admitted).heads → e<a.dims.K → j<a.dims.V →
      (realForwardEntry _ call.storage a.scale.value compiler before).2
        (call.storage.base .finalState+(shape a admitted).stateOffset h e j)=
      MathTarget.finalState (shape a admitted)
        (inputsOfHeap (originalHeap (shape a admitted) (0:ℝ) original (forwardObjects a))) Bool.true h e j) := by
  obtain ⟨hs,hd,hdev,_⟩ := returned_metadata call.storage .output call.output
  refine ⟨Contract.forwardCheck_iff a,rfl,hs,hd,hdev.trans call.device,?_,?_⟩
  · have result := real_forward_output _ call.storage a.scale.value compiler before
    rwa [preparation.inputs] at result
  · intro final
    have result := real_forward_optional_final _ call.storage a.scale.value compiler before final
    rwa [preparation.inputs] at result

theorem real_backward_optional_initial (c : Shape) (storage : Storage c) (scale : ℝ)
    (compiler : Compiler c tree scale seed rf terminal ri) (checkpoint : Bool) (later : Memory ℝ)
    (enabled : ri=Bool.true) :
    ∀ h e j,h<c.heads → e<c.K → j<c.V →
      (realBackwardEntry c storage scale compiler checkpoint later).2
        (storage.base .initialGradient+c.stateOffset h e j)=
      MathTarget.initialGradient c (inputsOfHeap (storage.binding.view 0 later)) scale h e j := by
  cases ri <;> simp only [Bool.false_eq_true] at enabled
  exact real_backward_initial c storage scale compiler checkpoint later

/-- Independent dense mathematical adjoints for the actual later autograd
call, including terminal cotangents and the optional initial-state gradient. -/
specification chunk_retention_ops_repaired_real_backward
    (a : Contract.Backward) (admitted : Contract.BackwardContract a)
    (fcall : ForwardCall a.forward admitted.1) (call : BackwardCall a admitted needsInitial)
    (compiler : Compiler (shape a.forward admitted.1) tree a.forward.scale.value
      a.forward.initial.isSome a.forward.final a.terminal.isSome (a.forward.initial.isSome && needsInitial))
    (original before raw later : Memory ℝ)
    (fpreparation : PreparedInputs (shape a.forward admitted.1) fcall.storage (0:ℝ)
      original before (forwardObjects a.forward))
    (preparation : PreparedInputs (shape a.forward admitted.1) call.storage (0:ℝ)
      raw later (backwardObjects a))
    (versions : Versions (shape a.forward admitted.1)
      (fcall.storage.binding.view 0 (realForwardEntry _ fcall.storage a.forward.scale.value compiler before).2)
      (call.storage.binding.view 0 later) (decide (a.forward.checkpoint=1))) :
    let c := shape a.forward admitted.1
    let result := realBackwardEntry c call.storage a.forward.scale.value compiler
      (decide (a.forward.checkpoint=1)) later
    let xs := Separate.joined (originalHeap c (0:ℝ) original (forwardObjects a.forward))
      (originalHeap c (0:ℝ) raw (backwardObjects a))
    (Contract.backwardCheck a=Bool.true ↔ Contract.BackwardContract a) ∧
    HostABI.LaunchABI c ∧ result.1=backwardTensors call.storage (a.forward.initial.isSome && needsInitial) ∧
    (∀ address,address.Valid c → address.region∈[Region.dq,Region.dk,Region.dv] →
      result.2 (call.storage.binding.address address)=realGradientTarget c xs a.forward.scale.value address) ∧
    ((a.forward.initial.isSome && needsInitial)=Bool.true → ∀ h e j,h<c.heads → e<c.K → j<c.V →
      result.2 (call.storage.base .initialGradient+c.stateOffset h e j)=
      MathTarget.initialGradient c (inputsOfHeap (originalHeap c (0:ℝ) raw (backwardObjects a)))
        a.forward.scale.value h e j) ∧
    (∀ point,(∀ s∈recomputeCode c tree (decide (a.forward.checkpoint=1))++
        backwardCode c tree (a.forward.initial.isSome && needsInitial),point≠call.storage.binding.address s.address) →
      result.2 point=later point) := by
  dsimp only
  refine ⟨Contract.backwardCheck_iff a,Contract.launchABI _ admitted.1.1,rfl,?_,?_,?_⟩
  · intro address valid tag
    have result := real_backward_gradients _ fcall.storage call.storage a.forward.scale.value compiler
      (decide (a.forward.checkpoint=1)) before later versions address valid tag
    simpa only [Separate.joined,fpreparation.inputs,preparation.inputs] using result
  · intro enabled
    have result := real_backward_optional_initial _ call.storage a.forward.scale.value compiler
      (decide (a.forward.checkpoint=1)) later enabled
    rwa [preparation.inputs] at result
  · intro point outside
    simp only [realBackwardEntry,compiler.backward_eq]
    exact code_frame realOps _ call.storage.binding _ _ later point outside

#axiomsClean real_backward_gradients
#axiomsClean real_backward_initial
#axiomsClean chunk_retention_ops_repaired_real_returns
#axiomsClean chunk_retention_ops_repaired_real_backward
end Checked
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
