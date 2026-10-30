import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelGrid
open VeriTile.Triton FP.Structural KernelSource KernelExecution
noncomputable section

def atProgram {α : Type} (s : State α) (pid : Nat) : State α :=
  { s with pids := Function.update s.pids 0 pid }

/-- A launch order executes the actual shared kernel. Empty launches leave
the complete state unchanged. Tensor ownership supplies schedule validity in
the public result; no wrapper or allocation execution is added here. -/
def run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) :
    List Nat → State α → Option (State α)
  | [], s => some s
  | pid :: tail, s => do
      let next ← FP.Structural.exec M (kernel p) (atProgram s pid)
      run M p tail next

theorem successful {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (order : List Nat) (s : State α) :
    ∃ final, run (backend M convert equal) p order s = some final := by
  induction order generalizing s with
  | nil => exact ⟨s, rfl⟩
  | cons pid tail ih =>
    obtain ⟨next, hn⟩ := execution M convert equal p (atProgram s pid) hw
    obtain ⟨final, hf⟩ := ih next
    refine ⟨final, ?_⟩
    change FP.Structural.exec (backend M convert equal) (kernel p)
      (atProgram s pid) = some next at hn
    simpa only [run, hn, Option.bind_some] using hf

#axiomsClean successful
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelGrid
