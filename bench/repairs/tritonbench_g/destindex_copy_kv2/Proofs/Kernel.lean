import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.Bridge

namespace VeriTile.Bench.Repairs.DestindexCopyKv2
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
abbrev touched := DestindexCopyKv1.touched
abbrev observeByte := DestindexCopyKv1.observeByte

namespace Mathematical
variable {α : Type} [Inhabited α]
def step (rest : Algebra α) (c : Config) (s : State α) (p : Nat) : State α :=
  (Tile.execution rest (layout c) {s with pids := fun d => if d=0 then p else s.pids d}).val
def run (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α) : State α :=
  ps.foldl (step rest c) s

theorem step_memory (rest : Algebra α) (c : Config) (width : block c.dims=c.dims)
    (s t : State α) (same : s.mem=t.mem) (p : Nat) :
    (step rest c s p).mem=(DestindexCopyKv1.Grid.step rest c t p).mem := by
  unfold step
  rw [same_source_memory rest _ width.symm]
  unfold DestindexCopyKv1.Grid.step
  rw [DestindexCopyKv1.Tile.execution_memory,DestindexCopyKv1.Tile.execution_memory]
  unfold DestindexCopyKv1.Tile.copied
  rw [VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_memory,
    VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_memory]
  simp only [DestindexCopyKv1.Tile.loaded,DestindexCopyKv1.Tile.inputOffset,
    DestindexCopyKv1.Tile.outputOffset,DestindexCopyKv1.Tile.destination,layout,
    DestindexCopyKv1.layout,ite_true,same]

theorem run_memory (rest : Algebra α) (c : Config) (width : block c.dims=c.dims)
    (ps : List Nat) (s t : State α) (same : s.mem=t.mem) :
    (run rest c ps s).mem=(DestindexCopyKv1.Grid.run rest c ps t).mem := by
  induction ps generalizing s t with
  | nil => exact same
  | cons p tail ih => exact ih _ _ (step_memory rest c width s t same p)

def execute (rest : Algebra α) (m : Inputs) (w : Nat → Word) (before : Nat → α) (other : State α) : Nat → α :=
  if Nonempty m then
    DestindexCopyKv1.project (base m .output) (m .output).numel before (fun x =>
      ((run rest (config m) (List.range (config m).count)
        (DestindexCopyKv1.initial (fun x => before (addr (base m .source) x))
          (fun x => before (addr (base m .output) x)) (destination w) other)).mem DestindexCopyKv1.outputRegion x).read .real)
  else before

theorem correct (rest : Algebra α) (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (before : Nat → α) (other : State α) :
    RowCopy (config m) (destination w) (fun x => before (addr (base m .source) x))
      (fun x => before (addr (base m .output) x))
      (fun x => execute rest m w before other (addr (base m .output) x)) ∧
    (∀ cell, (¬∃ x, touched (config m) (destination w) x ∧ cell=addr (base m .output) x) →
      execute rest m w before other cell=before cell) := by
  by_cases ne : Nonempty m
  · have eq : execute rest m w before other=DestindexCopyKv1.execute rest m w before other := by
      unfold execute
      rw [if_pos ne,run_memory rest (config m) (h.width ne) _ _ _ rfl]
      rfl
    rw [eq]
    exact DestindexCopyKv1.execute_correct rest m w h.common before other
  · simp only [execute,if_neg ne]
    refine ⟨⟨?_,fun _ _ => rfl⟩,fun _ _ => by trivial⟩
    intro p a b hp ha hb
    exact False.elim (ne ⟨by omega,by omega,by omega⟩)
end Mathematical

#axiomsClean Mathematical.step_memory
#axiomsClean Mathematical.correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv2
