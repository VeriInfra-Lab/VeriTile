import bench.repairs.tritonbench_g.cross_entropy1.Proofs.HostTrace

namespace VeriTile.Bench.Repairs.CrossEntropy1.Autograd
open Host Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

def savedTensors (context : Context) : List Tensor :=
  [context.logits,context.lse,context.labels,context.inplaceLogits]

structure Saved where
  context : Context
  version : Nat → Nat

/-- Ordinary saved-object/version metadata at the forward save boundary. -/
def capture {σ : Type} (versions : σ → Nat → Nat) (context : Context) (native : σ) : Saved :=
  ⟨context,versions native⟩

def Unmodified {α σ : Type} (versions : σ → Nat → Nat) (saved : Saved) (s : World α σ) : Prop :=
  ∀ t∈savedTensors saved.context, versions s.native t.objectId=saved.version t.objectId

/-- The first operation in Python backward retrieves ctx.saved_tensors.
Version failure therefore precedes upstream cloning, allocation, and launch. -/
def retrieve {α σ : Type} (versions : σ → Nat → Nat) (saved : Saved) (s : World α σ) : Step (World α σ) Context := by
  classical
  exact if Unmodified versions saved s then .ok saved.context s
    else .error "saved tensor version mismatch" s

def backwardEntry {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (saved : Saved) (upstream : Tensor) (s : World α σ) : Step (World α σ) Tensor :=
  bind (retrieve versions saved s) (fun context ready => Host.backward M rt context upstream ready)

theorem capture_context {σ : Type} (versions : σ → Nat → Nat) (context : Context) (native : σ) :
    (capture versions context native).context=context := rfl

theorem unmodified_entry {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (saved : Saved) (upstream : Tensor) (s : World α σ)
    (same : Unmodified versions saved s) :
    backwardEntry M rt versions saved upstream s=Host.backward M rt saved.context upstream s := by
  simp only [backwardEntry,retrieve,if_pos same,Host.bind]

theorem modified_entry {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (saved : Saved) (upstream : Tensor) (s : World α σ)
    (t : Tensor) (present : t∈savedTensors saved.context)
    (changed : versions s.native t.objectId≠saved.version t.objectId) :
    backwardEntry M rt versions saved upstream s=.error "saved tensor version mismatch" s := by
  have mismatch : ¬Unmodified versions saved s := fun h => changed (h t present)
  simp only [backwardEntry,retrieve,if_neg mismatch,Host.bind]

theorem backward_success {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (saved : Saved) (upstream output : Tensor) (s after : World α σ)
    (success : backwardEntry M rt versions saved upstream s=.ok output after) :
    Unmodified versions saved s ∧ Host.backward M rt saved.context upstream s=.ok output after := by
  by_cases same : Unmodified versions saved s
  · exact ⟨same,by simpa only [unmodified_entry M rt versions saved upstream s same] using success⟩
  · simp only [backwardEntry,retrieve,if_neg same,Host.bind] at success
    cases success

#axiomsClean capture_context
#axiomsClean unmodified_entry
#axiomsClean modified_entry
#axiomsClean backward_success
#specNonCircular capture avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#specNonCircular retrieve avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Autograd
