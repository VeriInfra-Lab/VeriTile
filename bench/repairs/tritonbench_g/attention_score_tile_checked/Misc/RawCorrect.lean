import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourceBridge
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentMachine

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.RawCorrect
open VeriTile Triton Reused Geometry Physical
noncomputable section

/-- Independent finite mathematical sum over logical Q/K/M arrays. -/
def kernelMathematical (c : Shape) (xs : LogicalInputs ℝ c) (offset size : Int)
    (scale : ℝ) (window complement : Bool) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : ℝ :=
  RealPhysical.logicalTarget c xs offset size scale window complement z h j

/-- Complete shared-source grid execution over raw word-indexed storage.
The ABI contains region bases, lengths, capacity and output/input separation;
read equations connect that storage to independent logical input arrays.
Machine bounds and the static tile bound are kernel formation assumptions. -/
theorem kernel_correct (c : Shape) (abi : ABI c) (physical : Nat → ℝ)
    (xs : LogicalInputs ℝ c) (reads : Normalized c abi physical xs)
    (offset size : Int) (scale : ℝ) (window complement : Bool)
    (domain : Whole002.Representation.Domain c abi offset size window) (tile : c.B*c.D≤1048576) :
    Spec.Real (Whole002.Machine.Closed c abi offset size window ∧
      (∀ shape ∈ statementListShapes (kernel c offset size scale window complement).surfaceBody,
        shape.prod≤1048576) ∧
      ∃ result, realExecute c abi physical offset size scale window complement=some result ∧
        (∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
          result ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)=
            kernelMathematical c xs offset size scale window complement z h j) ∧
        (∀ (r : RegionName), tracked r → r≠"out" → ∀ i, i<(abi.alloc r).length →
          result ((abi.alloc r).base+i)=physical ((abi.alloc r).base+i)) ∧
        (∀ a, ¬(abi.alloc "out").Contains a → result a=physical a)) := by
  obtain ⟨result,run,values,inputs,frame⟩ := RealPhysical.returned c abi physical xs reads
    offset size scale window complement (Whole002.Representation.Domain.key_offsets_fit c abi offset size window domain)
  refine ⟨Whole002.Machine.closed c abi offset size window domain,
    kernel_tiles_bounded c offset size scale window complement tile,result,?_,values,inputs,frame⟩
  simpa only [realExecute,Whole002.Execution.ExactReal.execute_eq] using run

#specNonCircular kernelMathematical avoiding [realExecute,VeriTile.Triton.exec]
#axiomsClean kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.RawCorrect
