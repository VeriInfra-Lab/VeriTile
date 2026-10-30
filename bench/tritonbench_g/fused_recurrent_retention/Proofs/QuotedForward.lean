import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForwardPrefix
import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForwardLoop
import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForwardFinal

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 16000

/-- Complete forward source execution, with its numerical carry, reduction,
actual stores and all final registers proved from initial memory. The prefix
state is an existential conclusion, never an external premise. -/
theorem forward_full_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (s : Shape) (b : InputBuffers) (plans : Plans) (seed final : Bool)
    (initial : State (Staged.Node α)) (hBK : 0<s.BK) (covered : plans.output.Covers s.BK)
    (hscale : initial.regs .real [] "scale" =
      some (fun _ => Staged.tagged .scale (a.scale s.DK))) :
    ∃ entry, ForwardInvariant a s b initial seed (launch initial) 0 entry ∧ entry.mem=initial.mem ∧
      FP.Structural.exec (Staged.algebra a plans)
        (Program.forward (layout s b) forwardRegions seed final) initial =
      some (forwardFinalResult a s b initial
        (forwardTail a s b plans initial seed (launch initial) 0 entry) seed final (launch initial)) := by
  obtain ⟨entry,he,hinv,hmem⟩ := forward_prefix_values a s b plans seed final initial hscale
  obtain ⟨hm,hlast⟩ := forward_range_values a s b plans initial seed final (launch initial)
    0 entry (by omega) hBK covered hinv
  have hf := forward_final_values a s b plans seed final initial _ (launch initial) hlast
  refine ⟨entry,hinv,hmem,?_⟩
  change run (Staged.algebra a plans) (Program.forward (layout s b) forwardRegions seed final).surfaceBody initial = _
  rw [Program.forward_split,run_append,run_append,he]
  simp only [Option.bind_some,run,step,bind]
  change ((range (Staged.algebra a plans) "_t" 0 s.T 1
    (Program.forwardParts (layout s b) forwardRegions seed final).body entry).bind
    (fun u => some u)).bind
    (fun u => run (Staged.algebra a plans)
      (Program.forwardParts (layout s b) forwardRegions seed final).after u) = _
  rw [hm]
  exact hf

#axiomsClean forward_full_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
