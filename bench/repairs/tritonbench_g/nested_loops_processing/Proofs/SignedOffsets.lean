import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.SignedMemory

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedExecution
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

/-- The actual signed source registers agree with the independently used natural
coordinate formulas through their exact integer embeddings. -/
theorem final_offsets {α : Type} [Inhabited α] (count cols : Nat) (s : State α) :
    (finalState count cols s).regs .int [2,2] "a_offsets" =
      some (signed (TypedExecution.offsets cols 28)) ∧
    (finalState count cols s).regs .int [2,2] "c_offsets" =
      some (signed (TypedExecution.offsets cols 48)) := by
  simp [finalState,outerLoopState,outerState,middleLoopState,middleState,
    middlePrefixState,innerLoopState,innerState,prefixState,
    advances,bump,initialA,initialC,TypedExecution.offsets,signed,Nat.add_assoc]
  constructor <;> funext i <;>
    simp [signed,bump,initialA,initialC,TypedExecution.offsets,address,Nat.add_assoc,Nat.mul_comm]

#axiomsClean final_offsets
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedExecution
