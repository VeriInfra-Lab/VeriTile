import bench.tritonbench_g.reversed_cumsum.Proofs.Mathematical_RealBody
import bench.tritonbench_g.reversed_cumsum.Proofs.Mathematical_RealInit
import bench.tritonbench_g.reversed_cumsum.Proofs.Mathematical_SignedRange
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Mathematical_SignedIndex

/-! Complete source execution, including the empty time dimension. Original
and candidate initialization are each executed, then every actual reverse
iteration establishes the same independent padded suffix and tagged frame. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealExecution
open VeriTile.Triton RealProgram RealStep RealLoop
open VeriTile.Bench.TritonBenchG.ReversedCumsumScalar
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem mirror_invariant (initial : BlockState) (p : Layout) (l : Launch)
    (s o : RegionName) (a : Nat → ℝ) (c : Nat) (u : BlockState)
    (h : Holds initial p l s o a c u) (j : Int) :
    Holds initial p l s o a c (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) := by
  refine ⟨h.undef, h.count_le, ?_, ?_, ?_, ?_, h.input, h.output, h.frame⟩
  · simpa using h.feature
  · simpa using h.head
  · simpa using h.triangular
  · simpa using h.accumulated

theorem mirror_assign (j : Int) (u : BlockState) :
    stepStmt (.assign .int [] "i_t"
      (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_i_t")))
      (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) =
    some ((u.setReg "__rev_i_t" .int [] (Tile.scalar j)).setReg "i_t" .int [] (Tile.scalar (-j))) := by
  simp only [stepStmt, evalOp, BlockState.setReg_same, Option.bind_some,
    Option.bind_eq_bind, Option.pure_def]
  apply congrArg some
  apply congrArg (fun tile : Tile .int [] =>
    (u.setReg "__rev_i_t" .int [] (Tile.scalar j)).setReg "i_t" .int [] tile)
  apply Tile.ext
  intro i
  change 0 - j = -j
  omega

theorem start_eval (p : Layout) (u : BlockState) :
    evalOp (reverseStart p) u = some (Tile.scalar (1 - (numChunks p : Int))) := by
  simp only [reverseStart, stopOp, evalOp, Option.bind_some,
    Option.bind_eq_bind, Option.pure_def]
  apply congrArg some
  apply Tile.ext
  intro i
  change 0 - ((((p.T + p.BT - 1) / p.BT : Nat) : Int) - 1) = 1 - (numChunks p : Int)
  unfold numChunks
  omega

def Post (p : Layout) (s o : RegionName) (initial final : BlockState) : Prop :=
  (∀ (t : Nat) (j : Fin p.BS), active p (RealInit.launch initial) t j.val →
    final.mem o (address p (RealInit.launch initial) t j.val) =
      MemCell.real (expected p (RealInit.launch initial) (initial.readMem s) (initial.undef s) t j)) ∧
  BlockState.WriteWithin (footprint p (RealInit.launch initial) o) initial final

end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealExecution
