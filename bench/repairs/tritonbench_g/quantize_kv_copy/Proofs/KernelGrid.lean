import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelGrid
open VeriTile.Triton FP.Structural Kernel
noncomputable section

def atProgram (p : Parameters) (s : State α) (pid : Nat) : State α :=
  {s with pids := Function.update (Function.update s.pids 0 (pid/p.heads)) 1 (pid%p.heads)}

/-- Each grid element executes the complete canonical two-dimensional kernel.
The flattened scheduling index is split into the actual token/head PIDs. -/
def run [Inhabited α] (op : Ops α) (p : Parameters) : List Nat → State α → Option (State α)
  | [],s => some s
  | pid::tail,s => do
      let next ← FP.Structural.exec (backend op) (kernel p) (atProgram p s pid)
      run op p tail next

theorem successful [Inhabited α] (op : Ops α) (p : Parameters) (order : List Nat) (s : State α) :
    ∃ final,run op p order s=some final := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl⟩
  | cons pid tail ih =>
    obtain ⟨next,hn⟩ := execution op p (atProgram p s pid)
    obtain ⟨final,hf⟩ := ih next
    exact ⟨final,by simpa only [run,hn,Option.bind_some] using hf⟩

#axiomsClean successful
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelGrid
