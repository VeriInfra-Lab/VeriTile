import bench.tritonbench_g.int4_matmul.Proofs.Kernel.GraphExecution
import bench.tritonbench_g.int4_matmul.Proofs.Kernel.GraphValues

namespace VeriTile.Bench.Repairs.Int4Matmul.SourceGraph
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem observe_write (p : Ops α) (s : FP.Structural.State (Node α))
    (region : RegionName) (offset : Nat) (cell : Cell (Node α)) :
    observeState p (s.write region offset cell) =
      (observeState p s).write region offset (mapCell (observe p) cell) := by
  simp only [observeState,FP.Structural.State.write]
  congr 1
  funext r i
  split <;> rfl

theorem writeLane_value (p : Ops α) (cfg : KernelSource.Parameters)
    (initial current : FP.Structural.State (Node α))
    (values : KernelExecution.CIndex cfg → α) (i : KernelExecution.CIndex cfg) :
    observeState p (KernelExecution.writeLane (ops p) cfg initial
      (fun j => .value (values j)) current i) =
      KernelExecution.writeLane p cfg (observeState p initial) values (observeState p current) i := by
  have active : KernelExecution.active cfg (observeState p initial) i = KernelExecution.active cfg initial i := rfl
  have address : KernelExecution.outputOffset cfg (observeState p initial) i =
      KernelExecution.outputOffset cfg initial i := rfl
  unfold KernelExecution.writeLane
  simp only [active,address]
  by_cases enabled : KernelExecution.active cfg initial i
  · simp only [enabled,ite_true,observe_write]
    by_cases split : cfg.splits = 1
    · simp only [split,ite_true,mapCell,mapValue,observe]
    · simp only [split,ite_false,ops,mapCell,mapValue,observe]
      change _ = (observeState p current).write _ _
        (.mk .real (p.atomicAdd ((mapCell (observe p) (current.mem _ _)).read .real) (values i)))
      rw [read_observe]
      rfl
  · simp only [enabled,ite_false]

theorem fold_value (p : Ops α) (cfg : KernelSource.Parameters)
    (initial current : FP.Structural.State (Node α)) (values : KernelExecution.CIndex cfg → α)
    (lanes : List (KernelExecution.CIndex cfg)) :
    observeState p (lanes.foldl (KernelExecution.writeLane (ops p) cfg initial
      (fun j => .value (values j))) current) =
      lanes.foldl (KernelExecution.writeLane p cfg (observeState p initial) values)
        (observeState p current) := by
  induction lanes generalizing current with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons,ih,writeLane_value]

theorem written_observe (p : Ops α) (cfg : KernelSource.Parameters)
    (initial : FP.Structural.State (Node α)) :
    observeState p (KernelExecution.written (ops p) cfg initial) =
      KernelExecution.written p cfg (observeState p initial) := by
  unfold KernelExecution.written
  have values := funext (accumulation_value p cfg initial (KernelSource.steps cfg))
  rw [values,fold_value]

/-- Reuse the already proved complete-source execution and exact raw Cell
frame, now for the unchanged standard Structural executor. -/
theorem source_memory (p : Ops α) (rest : Algebra α) (cfg : KernelSource.Parameters)
    (initial : FP.Structural.State (Node α)) :
    ∃ final, FP.Structural.exec (engine p rest) (KernelSource.kernel cfg) initial = some final ∧
      final.mem = (KernelExecution.written (ops p) cfg initial).mem := by
  simpa only [source_equal] using KernelExecution.correctness_memory (ops p) (base p rest) cfg initial

/-- Exact projection to the old independent fixed-K8/source-atomic reference.
All K masks, arbitrary strides, repeated output addresses and exterior cells
remain covered by the full-memory statement. -/
theorem source_memory_observe (p : Ops α) (rest : Algebra α) (cfg : KernelSource.Parameters)
    (initial : FP.Structural.State (Node α)) :
    ∃ final, FP.Structural.exec (engine p rest) (KernelSource.kernel cfg) initial = some final ∧
      final.mem = (KernelExecution.written (ops p) cfg initial).mem ∧
      (observeState p final).mem = (KernelExecution.written p cfg (observeState p initial)).mem := by
  obtain ⟨final,ran,memory⟩ := source_memory p rest cfg initial
  refine ⟨final,ran,memory,?_⟩
  rw [←written_observe]
  exact congrArg (fun m r i => mapCell (observe p) (m r i)) memory

#axiomsClean written_observe
#axiomsClean source_memory
#axiomsClean source_memory_observe
end
end VeriTile.Bench.Repairs.Int4Matmul.SourceGraph
