import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Autograd

namespace VeriTile.Bench.Repairs.CrossEntropy1.Autograd
open Host Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Exact Python backward tuple, including its seven trailing None entries.
Host.backward exposes the first gradient; this function restores the complete
Python return object at the custom-autograd boundary. -/
def pythonBackward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (saved : Saved) (upstream : Tensor) (before : World α σ) :
    Step (World α σ) (List (Option Tensor)) :=
  Host.bind (backwardEntry M rt versions saved upstream before)
    (fun output after => .ok [some output,none,none,none,none,none,none,none] after)

theorem python_return {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (saved : Saved) (upstream output : Tensor) (before after : World α σ)
    (success : backwardEntry M rt versions saved upstream before=.ok output after) :
    pythonBackward M rt versions saved upstream before=.ok [some output,none,none,none,none,none,none,none] after := by
  simp only [pythonBackward,success,Host.bind]

theorem python_failure {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (saved : Saved) (upstream : Tensor) (before failed : World α σ) (reason : String)
    (failure : backwardEntry M rt versions saved upstream before=.error reason failed) :
    pythonBackward M rt versions saved upstream before=.error reason failed := by
  simp only [pythonBackward,failure,Host.bind]

#axiomsClean python_return
#axiomsClean python_failure
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Autograd
