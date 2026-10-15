import bench.tritonbench_g.int4_matmul.DotAccCorrect
namespace VeriTile.Bench.TritonBenchG.Int4Matmul.FPDotAccWitness
open VeriTile.Triton FP.Structural FPTyped FPDotAcc FPDotAccGap
open FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- The very same full 64×128×32, two-block packed input from the prior
counterexample. No GPU result is asserted. Reversing FMA order changes bits. -/
theorem descending_bits :
    plannedOutput descendingPlan parameters initial lane = hword 0x4000 := by
  decide +kernel

theorem two_valid_plans_differ :
    plannedOutput ascendingPlan parameters initial lane = hword 0x0000 ∧
    plannedOutput descendingPlan parameters initial lane = hword 0x4000 := by
  exact ⟨by rw [ascending_output]; exact two_block_stored_bits_differ.2, descending_bits⟩

private theorem outside_output : ∀ i, active parameters initial i →
    (1 : Nat) ≠ outputOffset parameters initial i := by
  intro i hi
  have hv : i.1.val=0 ∧ i.2.1.val=0 := by
    simpa [active,rowBlock,columnBlock,firstRow,rowsInGroup,blocksM,blocksN,
      parameters,initial,Nat.lt_one_iff] using hi
  simp [outputOffset,rowBlock,columnBlock,firstRow,rowsInGroup,blocksM,blocksN,
    parameters,initial,hv.1,hv.2]

/-- Successful execution of the complete new parsed graph, including all
64 input products, arbitrary unused rows/columns, stored bits and frame. -/
theorem successful_complete_carry_graph (plan : FMAPlan) :
    ∃ final,
      FP.Structural.exec (backend (FPSoftware.scalar FPClosed.unused) (plannedDotAcc plan))
        (FPDotAcc.source parameters) initial = some final ∧
      final.mem "output" 0 = .mk .fp16 (plannedOutput plan parameters initial lane) ∧
      final.mem "output" 1 = initial.mem "output" 1 ∧
      final.mem = (plannedReference plan parameters initial).mem := by
  obtain ⟨final,he,hm,hp,_⟩ := int4_matmul_dotacc_plan_correctness plan
    parameters initial actual_scope_inputs.1 actual_scope_inputs.2.1 actual_scope_inputs.2.2 layout
  refine ⟨final,he,?_,?_,hm⟩
  · exact hp.1 lane (by decide)
  · exact hp.2 "output" 1 (Or.inr outside_output)

/-- Both plans are actual successful whole-program software executions.
The old original-source profile still stores half 1 on these same inputs. -/
theorem two_successful_graphs :
    ∃ a b,
      FP.Structural.exec (backend (FPSoftware.scalar FPClosed.unused) (plannedDotAcc ascendingPlan))
        (FPDotAcc.source parameters) initial = some a ∧
      FP.Structural.exec (backend (FPSoftware.scalar FPClosed.unused) (plannedDotAcc descendingPlan))
        (FPDotAcc.source parameters) initial = some b ∧
      a.mem "output" 0 = .mk .fp16 (hword 0x0000) ∧
      b.mem "output" 0 = .mk .fp16 (hword 0x4000) ∧
      FPClosed.outputBits parameters initial lane = hword 0x3c00 := by
  obtain ⟨a,ha,hav,_,_⟩ := successful_complete_carry_graph ascendingPlan
  obtain ⟨b,hb,hbv,_,_⟩ := successful_complete_carry_graph descendingPlan
  exact ⟨a,b,ha,hb,two_valid_plans_differ.1 ▸ hav,two_valid_plans_differ.2 ▸ hbv,
    two_block_stored_bits_differ.1⟩

#axiomsClean descending_bits
#axiomsClean two_valid_plans_differ
#axiomsClean successful_complete_carry_graph
#axiomsClean two_successful_graphs
end
end VeriTile.Bench.TritonBenchG.Int4Matmul.FPDotAccWitness
