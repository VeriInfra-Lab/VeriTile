import bench.tritonbench_g.swiglu_backward.Proofs.WholePublic

namespace VeriTile.Bench.Repairs.SwigluBackward.KernelRealResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
noncomputable section

/-- Complete raw source grid, independent analytic gradient/forward values,
and every exterior typed memory cell. -/
theorem swiglu_backward_kernel_correct (c : Layout) (B : Nat) (R : Bool)
    (positive : 0 < B) (own : Ownership c R) (tiles : List (Nat × Nat))
    (schedule : tiles.Perm (grid c B)) (before : State ℝ) :
    VeriTile.Spec.Real (∃ final, runTiles realAlgebra c B R tiles before = some final ∧
      (∀ ch, enabled R ch → ∀ r q, r < c.rows → q < c.cols →
        final.mem (c.output ch) (r*c.stride ch+q) = Cell.mk .real
          (formula ch ((before.mem c.X (r*c.sx+q)).read .real)
            ((before.mem c.Y (r*c.sy+q)).read .real)
            ((before.mem c.DOUT (r*c.sd+q)).read .real))) ∧
      (∀ region a, WholeOutside c R region a → final.mem region a = before.mem region a)) := by
  obtain ⟨final,ran,values,frame⟩ := grid_result realAlgebra c B R positive own tiles schedule before
  refine ⟨final,ran,?_,frame⟩
  intro ch on r q hr hq
  simpa only [scalar_real] using values ch on r q hr hq

#axiomsClean swiglu_backward_kernel_correct
#specNonCircular formula avoiding [Layout.source,runTiles]
end
end VeriTile.Bench.Repairs.SwigluBackward.KernelRealResult
