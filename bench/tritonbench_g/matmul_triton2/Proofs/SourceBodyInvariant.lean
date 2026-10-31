import bench.tritonbench_g.matmul_triton2.Proofs.SourceBodyExecution
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton2.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem metadata_unchanged (p : Core.Parameters) (initial current : State (Word α))
    (block : Nat) (name : RegName) (n : Nat)
    (member : (name,n) ∈ VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters p) :
    (bodyResult p initial current block).regs .nat [] name = current.regs .nat [] name := by
  simp only [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with h | h | h | h | h | h | h | h | h
  all_goals cases h
  all_goals simp [bodyResult]

theorem bodyInvariant (p : Core.Parameters) (initial current : State (Word α)) (block : Nat)
    (h : RuntimeInvariant p initial block current) :
    RuntimeInvariant p initial (block+1) (bodyResult p initial current block) := by
  constructor
  · simpa [bodyResult] using h.memory
  · simpa [bodyResult] using h.pids
  · intro name n member
    rw [metadata_unchanged p initial current block name n member]
    exact h.metadata name n member
  all_goals simp [bodyResult,h.offm,h.offn,h.offk,Core.aOffset,Core.bOffset,
    Nat.add_mul,Nat.add_assoc]
  all_goals funext i
  all_goals simp [Core.aOffset,Core.bOffset,Nat.add_mul,Nat.add_assoc]

#axiomsClean metadata_unchanged
#axiomsClean bodyInvariant
end
end VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
