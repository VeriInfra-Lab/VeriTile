import bench.tritonbench_g.quantize_copy_kv.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelGrid
open VeriTile.Triton FP.Structural KernelSource KernelExecution
noncomputable section

def atProgram {α : Type} (s : State α) (pid : Nat) : State α :=
  { s with pids := Function.update s.pids 0 pid }

/-- A launch order executes the actual shared kernel. Empty launches leave
the complete state unchanged. Tensor ownership supplies schedule validity in
the public result; no wrapper or allocation execution is added here. -/
def run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (version : Bool) :
    List Nat → State α → Option (State α)
  | [], s => some s
  | pid :: tail, s => do
      let next ← FP.Structural.exec M (kernel version p) (atProgram s pid)
      run M p version tail next

theorem same_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (order : List Nat) (s : State α) : run M p Bool.false order s = run M p Bool.true order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    simp only [run]
    have hs := KernelExecution.same_run M p (atProgram s pid)
    change FP.Structural.exec M (kernel Bool.false p) _ =
      FP.Structural.exec M (kernel Bool.true p) _ at hs
    rw [hs]
    cases FP.Structural.exec M (kernel Bool.true p) (atProgram s pid) with
    | none => rfl
    | some next => exact ih next

theorem successful {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (order : List Nat) (s : State α) :
    ∃ final, run (backend M convert equal) p Bool.false order s = some final := by
  induction order generalizing s with
  | nil => exact ⟨s, rfl⟩
  | cons pid tail ih =>
    obtain ⟨next, hn⟩ := execution M convert equal p (atProgram s pid) hw
    obtain ⟨final, hf⟩ := ih next
    refine ⟨final, ?_⟩
    change FP.Structural.exec (backend M convert equal) (kernel Bool.false p)
      (atProgram s pid) = some next at hn
    simpa only [run, hn, Option.bind_some] using hf

theorem successful_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (convert : α → Int) (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (order : List Nat) (s : State α) :
    ∃ final, run (backend M convert equal) p Bool.false order s = some final ∧
      run (backend M convert equal) p Bool.true order s = some final := by
  obtain ⟨final, hf⟩ := successful M convert equal p hw order s
  exact ⟨final, hf, (same_run _ p order s).symm.trans hf⟩

#axiomsClean same_run
#axiomsClean successful
#axiomsClean successful_equivalence
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelGrid
