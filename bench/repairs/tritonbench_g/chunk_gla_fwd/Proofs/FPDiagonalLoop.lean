import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPDiagonalValues
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def enabled (split : Bool) (c : Layout) (s : State α) : Prop := rowStart split c s<c.T
instance (split : Bool) (c : Layout) (s : State α) : Decidable (enabled split c s) := inferInstanceAs (Decidable (_<_))
def written (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ) (s : State α) : State α :=
  foldRange (columnWritten M split c scale s) 0 (columns split c s) 1 s

theorem loopExecution (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (initial current : State α) (ctx : Context M split c initial current 0) (hmem : current.mem=initial.mem) :
    ∃ t, range (engine M) "j" 0 (columns split c initial) 1 (parts split c scale).body current=some t ∧
      t.mem=(written M split c scale initial).mem := by
  let I := fun (n : Nat) (a : State α) (s : State α) => Context M split c initial s n ∧ s.mem=a.mem
  exact range_values (engine M) "j" (parts split c scale).body (columnWritten M split c scale initial)
    I (fun a s => s.mem=a.mem) 0 (columns split c initial) 1 initial current
    (by
      intro j a s h hj
      let t := bodyExecution M split c scale initial s j h.1
      refine ⟨t.val,t.property,bodyContext M split c scale initial s j h.1,?_⟩
      exact (bodyMemory M split c scale initial s j h.1).trans
        (column_memory M split c scale initial j s a h.2))
    (by intro n a s h; exact h.2) ⟨ctx,hmem⟩

#axiomsClean loopExecution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
