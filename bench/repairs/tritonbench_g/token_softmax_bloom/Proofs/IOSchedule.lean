import bench.repairs.tritonbench_g.token_softmax_bloom.IO

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem)
open SoftmaxIO
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
variable {α : Type} [Inhabited α]

/-- Reuse the source's arbitrary-state execution theorem. The source reads no
unbound register, so its memory result equals the established grid launch. -/
theorem one (M : Algebra α) (p : Parameters) (positive : 0 < p.width)
    (pid : Grid.PID) (s : State α) :
    ∃ t, (invocation M p pid).run s = some t ∧
      t.mem = (reference M .fp32 p (Grid.state s.mem pid)).mem := by
  refine ⟨(execution M .fp32 p ((invocation M p pid).initial s) positive).val,
    (execution M .fp32 p ((invocation M p pid).initial s) positive).property, ?_⟩
  rw [execution_memory]
  have rows : inputRow M .fp32 p ((invocation M p pid).initial s) =
      inputRow M .fp32 p (Grid.state s.mem pid) := by funext i; rfl
  have counts : count p ((invocation M p pid).initial s) = count p (Grid.state s.mem pid) := rfl
  have offsets : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.outputOffset p
      ((invocation M p pid).initial s) =
      VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.outputOffset p (Grid.state s.mem pid) := by
    funext i; rfl
  simp only [reference, fold_writes_mem, outputValues, numerators, rows, counts, offsets,
    Invocation.initial_mem]
  rfl

/-- Whole invocation list, preserving exactly the existing memory-level grid
execution while keeping its actual final Structural state. -/
theorem complete (M : Algebra α) (p : Parameters) (positive : 0 < p.width)
    (work : List Grid.PID) (s : State α) :
    ∃ t, runSchedule (work.map (invocation M p)) s = some t ∧
      t.mem = Grid.graph M p work s.mem := by
  induction work generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons pid rest ih =>
    obtain ⟨u,executed,memory⟩ := one M p positive pid s
    obtain ⟨t,restRun,restMemory⟩ := ih u
    refine ⟨t, ?_, ?_⟩
    · simpa only [List.map_cons, runSchedule, executed, Option.bind_some] using restRun
    · simpa only [Grid.graph, memory] using restMemory

#axiomsClean one
#axiomsClean complete
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOSchedule
