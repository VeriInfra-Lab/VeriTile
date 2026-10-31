import bench.tritonbench_g.chunk_gla_simple.Sources.FPExecution

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.FPValue
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def written {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial : State α) : State α :=
  (TileShape.allIndices [c.BT,c.BV]).foldl (fun t i => if active c initial i then
    t.write "o" (oOffset c initial i) (.mk .fp32 (outputValue M c d initial scale i)) else t) initial

def Post {α : Type} (c : Shape) (initial final : State α) (v : OIndex c → α) : Prop :=
  (∀ i, active c initial i → final.mem "o" (oOffset c initial i)=.mk .fp32 (v i)) ∧
  (∀ (r : RegionName) o, (r≠"o" ∨ ∀ i, active c initial i → o≠oOffset c initial i) → final.mem r o=initial.mem r o)

theorem readback {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial : State α) (i : OIndex c) (hi : active c initial i)
    (hw : ∀ i j, active c initial i → active c initial j → oOffset c initial i=oOffset c initial j → i=j) :
    (written M c scale d initial).mem "o" (oOffset c initial i)=.mk .fp32 (outputValue M c d initial scale i) :=
  masked_readback_active _ _ _ _ _ i hi hw

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial : State α) (r : RegionName) (off : Nat)
    (h : r≠"o" ∨ ∀ i, active c initial i → off≠oOffset c initial i) :
    (written M c scale d initial).mem r off=initial.mem r off := by
  apply State.masked_scatter_frame
  rcases h with hr | ho
  · exact Or.inl hr
  · right
    intro i _ hi
    exact ho i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.FPValue
