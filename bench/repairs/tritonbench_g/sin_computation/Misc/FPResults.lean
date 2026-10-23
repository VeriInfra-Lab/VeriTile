import bench.repairs.tritonbench_g.sin_computation.Proofs.Rewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.FPResults
open VeriTile.Triton
noncomputable section

/-- Complete successful execution of both sources and every untouched cell,
expressed directly over raw typed memory and the selected scalar backend. -/
theorem kernel_results {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (X O : RegionName) (n : Nat) (s : FP.Structural.State α) :
    ∃ final,
      FP.Structural.exec (FP.ProfiledRewrite.engine M) (kernel X O n blockSize) s=some final ∧
      FP.Structural.exec (FP.ProfiledRewrite.engine M) (indexCommutedKernel X O n blockSize) s=some final ∧
      (∀ (r : RegionName) address,
        (r≠O ∨ ∀ i : TileIndex [blockSize],s.pids 0*blockSize+i.1.val<n →
          address≠s.pids 0*blockSize+i.1.val) → final.mem r address=s.mem r address) := by
  let result:=Rewrite.originalExecution M X O n s
  refine ⟨result.val,result.property,
    (Rewrite.execution_eq M X O n s).symm.trans result.property,?_⟩
  intro r address outside
  dsimp only [result,Rewrite.originalExecution]
  apply FP.Structural.State.masked_scatter_frame
  exact outside.imp id (fun h i _ hi=>h i hi)

#axiomsClean kernel_results
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.FPResults
