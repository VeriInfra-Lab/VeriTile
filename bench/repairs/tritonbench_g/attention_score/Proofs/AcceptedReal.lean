import bench.repairs.tritonbench_g.attention_score.Proofs.CurrentExecution
import bench.repairs.tritonbench_g.attention_score.Proofs.CurrentMachine

/-! Public exact-real correctness for the private repaired FP32 candidate.
The target uses Q:[Z,H,N,D], K:[Z,HK,NK,D], supplied M:[Z,H,R] with R≥N,
positive HK dividing H, and power-of-two D≥16. The explicit Domain additionally checks actual launch
metadata, padded int32 block coordinates, signed64 byte displacements and
unsigned64 formed addresses; empty returns skip launch constraints. Input
normalization is an ordinary logical-value copy contract. Actual selected
blocks include the persistent single-retry 64/32/16 behavior. -/
namespace VeriTile.Bench.Repairs.AttentionScore.Whole002.Correct
open VeriTile Triton Geometry Physical
noncomputable section

theorem repaired_return (c : Shape) (dispatch : Dispatch c) (abi : ABI c) (physical : Nat → ℝ)
    (xs : LogicalInputs ℝ c) (normalized : Normalized c abi physical xs)
    (offset size : Int) (window complement : Bool)
    (domain : Representation.Domain c abi offset size window) :
    Machine.Closed c abi offset size window ∧ persistentBlock dispatch=c.B ∧
    ∃ result, Execution.ExactReal.execute c abi physical offset size (LogicalFormula.hostScale c.D) window complement=some result ∧
      (∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        result ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)=
          RealPhysical.logicalTarget c xs offset size (LogicalFormula.hostScale c.D) window complement z h j) ∧
      (∀ (r : RegionName), tracked r → r≠"out" → ∀ i, i<(abi.alloc r).length →
        result ((abi.alloc r).base+i)=physical ((abi.alloc r).base+i)) ∧
      (∀ a, ¬(abi.alloc "out").Contains a → result a=physical a) := by
  rw [Execution.ExactReal.execute_eq]
  exact ⟨Machine.closed c abi offset size window domain,persistent_selected dispatch,RealPhysical.returned c abi physical xs normalized offset size
    (LogicalFormula.hostScale c.D) window complement⟩

theorem all_programs_safe (c : Shape) (s : BlockState) (z : Fin c.Z) (h : Fin c.H)
    (block : Fin c.programs) (offset size : Int) (window complement : Bool)
    (abi : ABI c) (domain : Representation.Domain c abi offset size window) :
    (Execution.kernel c offset size (LogicalFormula.hostScale c.D) window complement).eraseDType.toAlgKernel.TraceSafe
      (DenseSafety.bounds c) (DenseSource.programState c s block.val z.val h.val) := by
  rw [Execution.kernel_eq]
  exact DenseSafety.source_safe c s z h block offset size (LogicalFormula.hostScale c.D) window complement

#axiomsClean repaired_return
#axiomsClean all_programs_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScore.Whole002.Correct
