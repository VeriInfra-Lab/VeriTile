import bench.repairs.tritonbench_g.adam_update_triton.IO
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Tile
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Math
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Math

namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Real
open VeriTile.Triton FP.Structural
noncomputable section
open Classical

theorem scalar_target (params : Scalars ℝ) (output : Output) (p g m : ℝ) :
    Tile.scalar semantics params output p g m=Math.target params output p g m := by
  rw [←Math.source_scalar_correct params output p g m]
  cases output <;> simp [Tile.scalar,semantics,
    VeriTile.Bench.Repairs.SwigluTriton.Math.algebra,Math.sourceScalar]

theorem program_values (params : Scalars ℝ) (total chunk block pid : Nat)
    (s : State ℝ) (output : Output) (lane : TileIndex [block])
    (active : pid*block+lane.1.val<chunk) :
    (Tile.sourceRun semantics params total chunk block pid s).val.mem
      (Tile.outputRegion output) (pid*block+lane.1.val)=
      Cell.mk .real (Math.target params output
        ((s.mem "P" (pid*block+lane.1.val)).read .real)
        ((s.mem "G" (pid*block+lane.1.val)).read .real)
        ((s.mem "M" (pid*block+lane.1.val)).read .real)) := by
  rw [Tile.readback semantics params total chunk block pid s output lane active,scalar_target]

/-- Operational assembly for the complete source program. The public IO
contract supplies the input windows and mathematical function separately. -/
theorem kernel_execution (params : Scalars ℝ) (total n B pid : Nat)
    (s : State ℝ) :
    ∃ final,
      run semantics.backend (source total n B).surfaceBody
        (Tile.programState params pid s) = some final ∧
      (∀ output (j : TileIndex [B]), pid * B + j.1.val < n →
        final.mem (Tile.outputRegion output) (pid * B + j.1.val) =
          Cell.mk .real (Math.target params output
            ((s.mem "P" (pid * B + j.1.val)).read .real)
            ((s.mem "G" (pid * B + j.1.val)).read .real)
            ((s.mem "M" (pid * B + j.1.val)).read .real))) ∧
      (∀ r o, Tile.Outside n B pid r o → final.mem r o = s.mem r o) := by
  exact ⟨(Tile.sourceRun semantics params total n B pid s).val,
    (Tile.sourceRun semantics params total n B pid s).property,
    program_values params total n B pid s,
    Tile.frame semantics params total n B pid s⟩

/-- Bind independent lane arrays to the three active input windows. Inactive
lanes remain unconstrained, and both in-place stores read the entry values. -/
theorem kernel_inputs (params : Scalars ℝ) (total n B pid : Nat)
    (s : State ℝ) (ps gs ms : Fin B → ℝ)
    (hp : ∀ j : Fin B, pid * B + j.val < n →
      (s.mem "P" (pid * B + j.val)).read .real = ps j)
    (hg : ∀ j : Fin B, pid * B + j.val < n →
      (s.mem "G" (pid * B + j.val)).read .real = gs j)
    (hm : ∀ j : Fin B, pid * B + j.val < n →
      (s.mem "M" (pid * B + j.val)).read .real = ms j) :
    ∃ final,
      run semantics.backend (source total n B).surfaceBody
        (Tile.programState params pid s) = some final ∧
      (∀ output (j : Fin B), pid * B + j.val < n →
        final.mem (Tile.outputRegion output) (pid * B + j.val) =
          Cell.mk .real (Math.target params output (ps j) (gs j) (ms j))) ∧
      (∀ r o, Tile.Outside n B pid r o → final.mem r o = s.mem r o) := by
  obtain ⟨final, execution, values, frame⟩ := kernel_execution params total n B pid s
  refine ⟨final, execution, ?_, frame⟩
  intro output j active
  simpa only [hp j active, hg j active, hm j active] using
    values output (j, PUnit.unit) active

#axiomsClean scalar_target
#axiomsClean program_values
#axiomsClean kernel_execution
#axiomsClean kernel_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Real
