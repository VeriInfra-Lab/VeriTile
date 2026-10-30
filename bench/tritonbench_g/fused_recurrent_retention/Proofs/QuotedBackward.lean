import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedBackwardPrefix
import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedBackwardFirstLoop
import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedBackwardMiddle
import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedReverseLoop

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

private theorem single_range {α : Type} [Inhabited α] (A : Algebra α)
    (name : RegName) (stop : Nat) (body : List ComputeStmt) (u : State α) :
    run A [.forRange name 0 stop 1 body] u=range A name 0 stop 1 body u := by
  simp only [run,step,bind]
  cases range A name 0 stop 1 body u <;> rfl

/-- Both loops and the repaired final-cotangent branch execute in the shared
source interpreter. Every intermediate is produced, and the reverse initial
memory equals the memory after the actual dQ stores. -/
theorem backward_full_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (s : Shape) (b : InputBuffers) (plans : Plans) (seed final : Bool)
    (initial : State (Staged.Node α)) (hBK : 0<s.BK) (hBV : 0<s.BV)
    (coveredQ : plans.dq.Covers s.BV) (coveredK : plans.dk.Covers s.BV) (coveredV : plans.dv.Covers s.BK)
    (hscale : initial.regs .real [] "scale" =
      some (fun _ => Staged.tagged .scale (a.scale s.DK))) :
    ∃ entry middle,
      BackwardFirstInvariant a s b initial seed (launch initial) 0 entry ∧ entry.mem=initial.mem ∧
      BackwardSecondInvariant a s b initial final (launch initial) 0 middle ∧
      middle.mem=(backwardFirstTail a s b plans initial seed (launch initial) 0 entry).mem ∧
      FP.Structural.exec (Staged.algebra a plans)
        (Program.backward (layout s b) backwardRegions seed final) initial =
      some (reverseTail a s b plans initial final (launch initial) 0 middle) := by
  obtain ⟨entry,he,hinv,hmem⟩ := backward_prefix_values a s b plans seed final initial hscale
  obtain ⟨hfirst,hfirstInv⟩ := backward_first_range_values a s b plans initial seed final (launch initial)
    0 entry (by omega) hBV coveredQ hinv
  obtain ⟨middle,hm,hmInv,hmMem⟩ := backward_middle_values a s b plans seed final initial _ (launch initial) hfirstInv
  obtain ⟨hsecond,hsecondInv⟩ := reverse_range_values a s b plans initial seed final (launch initial)
    0 middle (by omega) hBK hBV coveredK coveredV hmInv
  have hloop1 : run (Staged.algebra a plans)
      [.forRange "i" 0 (layout s b).T 1
        (Program.backwardFirst (layout s b) backwardRegions seed final).body] entry =
      some (backwardFirstTail a s b plans initial seed (launch initial) 0 entry) := by
    rw [single_range]
    exact hfirst
  have hloop2 : run (Staged.algebra a plans)
      [.forRange "_t" 0 (layout s b).T 1
        (Program.backwardSecond (layout s b) backwardRegions seed final).body] middle =
      some (reverseTail a s b plans initial final (launch initial) 0 middle) := by
    rw [single_range]
    exact hsecond
  refine ⟨entry,middle,hinv,hmem,hmInv,hmMem,?_⟩
  change run (Staged.algebra a plans) (Program.backward (layout s b) backwardRegions seed final).surfaceBody initial = _
  rw [Program.backward_split,run_append,run_append,run_append,he]
  simp only [Option.bind_some,hloop1,hm,hloop2]

#axiomsClean backward_full_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
