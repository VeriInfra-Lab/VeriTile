import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.KernelInterface
import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Ideal

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast
noncomputable section

theorem integer_grid_real (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (m : Memory ℝ ℝ) (divisor : ℝ) (bias : Bool)
    (events : List (Commit s c)) :
    gridRun Quote.integerRealOps s c b selected divisor bias events m=
      gridRun realOps s c b selected divisor bias events m := by
  induction events generalizing m with
  | nil=>rfl
  | cons e es ih=>
    rw [gridRun,Quote.integer_partial_real,gridRun,ih]
    rfl

namespace RealKernelResult
/-- Complete quoted commit execution on raw storage, independent dense Real
matmul/dequantization and every non-output element. -/
theorem kernel_real (s : Shape) (c : Config) (b : Buffers)
    (admitted : Source.Admitted c) (fresh : b.Fresh) (selected : Selection s c)
    (bias : Bool) (events : List (PhysicalCommit s c))
    (order : (events.map PhysicalCommit.logical).Perm (allCommits s c))
    (initial : Memory ℝ ℝ) (zero : Zeroed realOps s c b initial) :
    ∃ final,Quote.sourceGrid (Quote.idealInteger s c selected) s c bias b (1/16129) events initial=some final ∧
      (∀ cell : MatrixCell s,final (outputAddress s b cell)=
        .inr (.inr (target s (idealInput s b bias initial) cell.1 cell.2))) ∧
      (∀ address : Address,(∀ cell : MatrixCell s,address≠outputAddress s b cell) →
        final address=initial address) := by
  let q := Schedule.ofPermutation s c (events.map PhysicalCommit.logical) order
  refine ⟨gridRun realOps s c b selected (1/16129) bias q.events initial,?_,?_,?_⟩
  · rw [Quote.sourceGrid_eq _ _ _ _ _ _ selected rfl]
    change some (gridRun Quote.integerRealOps s c b selected (1/16129) bias
      (events.map PhysicalCommit.logical) initial)=_
    rw [integer_grid_real]
    rfl
  · intro cell
    rw [KernelInterface.output_word realOps s c b fresh selected (1/16129) bias q initial zero cell,
      real_reference]
  · exact fun address outside => grid_frame realOps s c b selected (1/16129) bias q.events initial address outside

#axiomsClean kernel_real
#specNonCircular target avoiding [Quote.sourceGrid,gridRun,partialValue]
end RealKernelResult
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast
