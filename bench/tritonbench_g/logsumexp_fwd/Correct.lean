import bench.tritonbench_g.logsumexp_fwd.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.LogsumexpFwd.Correct
open VeriTile.Triton FP.Structural
open scoped BigOperators LogicalProgramIO
noncomputable section

/-- Each logical row/block scratch result is a log of its filtered exponential
sum. The final Torch row aggregation remains outside this source contract. -/
specification logsumexp_kernel_correctness (a : RawKernel.Geometry) (scale : ℝ)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (a.rows*a.blocks))) :
    Spec.Real (ReaderIO.view a scale jobs ⊨{ReaderIO.realInputs a} fun xs row segment =>
      (Real.log (∑ i ∈ Finset.univ.filter
        (fun i : Fin a.block => segment.val * a.block + i.val < a.columns),
        Real.exp (if a.hasScale then
          ReaderIO.realAt xs row (segment.val * a.block + i.val) * scale
        else ReaderIO.realAt xs row (segment.val * a.block + i.val))) : WithBot ℝ)) :=
  ReaderProof.correct a scale jobs schedule

#axiomsClean logsumexp_kernel_correctness
end
end VeriTile.Bench.Repairs.LogsumexpFwd.Correct
