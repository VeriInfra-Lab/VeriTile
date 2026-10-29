import bench.tritonbench_g.swiglu_backward.Proofs.Observed

namespace VeriTile.Bench.Repairs.SwigluBackward.KernelNativeResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward
open Whole
noncomputable section

/-- Complete raw typed source execution with independent ordered native output
words. The original local compiler lowering and primitive model are unchanged. -/
theorem swiglu_backward_kernel_fp32 (M : Observed.NativeModel (BitVec 32))
    (c : Layout) (B : Nat) (R : Bool) (positive : 0 < B) (own : Ownership c R)
    (tiles : List (Nat × Nat)) (schedule : tiles.Perm (grid c B))
    (before : State (Observed.Term (BitVec 32)))
    (x y d : Nat → Nat → BitVec 32)
    (hx : ∀ r q, r < c.rows → q < c.cols →
      (before.mem c.X (r*c.sx+q)).read .real = .input (x r q))
    (hy : ∀ r q, r < c.rows → q < c.cols →
      (before.mem c.Y (r*c.sy+q)).read .real = .input (y r q))
    (hd : ∀ r q, r < c.rows → q < c.cols →
      (before.mem c.DOUT (r*c.sd+q)).read .real = .input (d r q)) :
    ∃ final, runTiles Observed.graphAlgebra c B R tiles before = some final ∧
      (∀ ch, enabled R ch → ∀ r q, r < c.rows → q < c.cols →
        Observed.interpret M ((final.mem (c.output ch) (r*c.stride ch+q)).read .real) =
          Observed.reference M ch (x r q) (y r q) (d r q)) ∧
      (∀ region a, WholeOutside c R region a → final.mem region a = before.mem region a) := by
  obtain ⟨final,ran,values,frame⟩ := grid_result Observed.graphAlgebra c B R positive own tiles schedule before
  refine ⟨final,ran,?_,frame⟩
  intro ch on r q hr hq
  rw [values ch on r q hr hq]
  rw [Cell.read_same, hx r q hr hq, hy r q hr hq, hd r q hr hq]
  exact Observed.source_graph_lowering M ch (x r q) (y r q) (d r q)

#axiomsClean swiglu_backward_kernel_fp32
#specNonCircular Observed.reference avoiding [Observed.interpret,scalar,runTiles]
end
end VeriTile.Bench.Repairs.SwigluBackward.KernelNativeResult
