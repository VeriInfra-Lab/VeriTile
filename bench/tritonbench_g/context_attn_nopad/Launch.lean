import bench.tritonbench_g.context_attn_nopad.FPBackend

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
namespace FPHost
 def atProgram {α : Type} (c : Call) (idx : GridIndex c.grid) (s : State α) : State α :=
  {s with pids := idx.toPids, numPids := c.grid.toNumPids}
 def jobs (c : Call) : List (GridIndex c.grid) := Finset.univ.toList
 theorem jobs_complete (c : Call) : (jobs c).Nodup ∧ ∀ idx : GridIndex c.grid, idx ∈ jobs c :=
  ⟨Finset.nodup_toList _,by simp [jobs]⟩
end FPHost
namespace FPHostSource
 structure Metadata {α : Type} [Inhabited α] (c : Call) (r : Regions) (s : State α) : Prop where
  start : ∀ b,b < c.batches → (s.mem r.start b).read .int = (c.start b : Int)
  length : ∀ b,b < c.batches → (s.mem r.sequence b).read .int = (c.length b : Int)
end FPHostSource
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
