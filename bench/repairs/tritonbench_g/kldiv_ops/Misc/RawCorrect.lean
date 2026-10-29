import bench.repairs.tritonbench_g.kldiv_ops.Proofs.RealGrid
import bench.repairs.tritonbench_g.kldiv_ops.Misc.RealHost
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.BackwardCoverage
import bench.repairs.tritonbench_g.kldiv_ops.Kernels

namespace VeriTile.Bench.Repairs.KldivOps.Kernel.RawCorrect
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete source grid, independent Real loss at every output, and the
entire typed-memory complement. Empty rows and columns remain supported. -/
theorem forward (p : Parameters) (flag : Bool) (red : Nat) (eps : ℝ)
    (s : State ℝ) (block : 0<p.block) (predictionSeparate : p.Y≠p.O)
    (targetSeparate : p.T≠p.O) :
    ∃ final, Program.grid RealExecution.algebra p flag red eps s=some final ∧
      (∀ row,row<p.rows → ∀ col,col<p.cols → red=0 →
        (final.mem p.O (row*p.cols+col)).read .real=
          RealExecution.loss flag eps ((s.mem p.Y (row*p.cols+col)).read .real)
            ((s.mem p.T (row*p.cols+col)).read .real)) ∧
      (∀ row,row<p.rows → red≠0 → (final.mem p.O row).read .real=
        RealExecution.rowLoss flag eps (fun i => (s.mem p.Y (row*p.cols+i)).read .real)
          (fun i => (s.mem p.T (row*p.cols+i)).read .real) p.cols) ∧
      (∀ r i,(∀ row,row<p.rows → ¬Program.Writes p red row r i) →
        final.mem r i=s.mem r i) := by
  obtain ⟨final,executed,elements,rows⟩ :=
    RealExecution.complete_grid p flag red eps s block predictionSeparate targetSeparate
  obtain ⟨other,otherExecuted,memory⟩ := Program.grid_complete RealExecution.algebra p flag red eps s block
  have same : other=final := Option.some.inj (otherExecuted.symm.trans executed)
  subst other
  refine ⟨final,executed,elements,rows,?_⟩
  intro r i outside
  rw [memory]
  exact Program.grid_frame _ p flag red eps _ s.mem r i
    (fun row member => outside row (List.mem_range.mp member))

/-- Complete backward source grid and its independent unscaled derivative.
Every cell outside the full destination row spans is preserved. -/
theorem backward (p : Parameters) (flag : Bool) (s : State ℝ)
    (block : 0<p.block) (targetSeparate : p.T≠p.O) :
    ∃ final, Backward.grid RealExecution.algebra p flag s=some final ∧
      (∀ row,row<p.rows → ∀ col,col<p.cols →
        (final.mem p.O (row*p.cols+col)).read .real=
          (if flag then -Real.exp ((s.mem p.T (row*p.cols+col)).read .real)
           else -((s.mem p.T (row*p.cols+col)).read .real))) ∧
      (∀ r i,(∀ row,row<p.rows → r ≠ p.O ∨ i < row*p.cols ∨ row*p.cols+p.cols ≤ i) →
        final.mem r i=s.mem r i) := by
  obtain ⟨final,executed,memory⟩ := Backward.grid_complete RealExecution.algebra p flag s block
  refine ⟨final,executed,?_,?_⟩
  · intro row hr col hc
    rw [memory,Backward.grid_value _ p flag _ s.mem row col (List.mem_range.mpr hr) block hc targetSeparate]
    simpa only [Cell.read_same] using RealExecution.backward_scalar_ideal flag
      ((s.mem p.T (row*p.cols+col)).read .real)
  · intro r i outside
    rw [memory]
    exact Backward.grid_frame _ p flag _ s.mem r i
      (fun row member => outside row (List.mem_range.mp member))

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.KldivOps.Kernel.RawCorrect
