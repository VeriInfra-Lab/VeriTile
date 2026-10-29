import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.PairIO

/-! The actual candidate removes mask restoration only from the unscaled
specialization, where masked FP32 loads already supply the same negative
infinity. All reductions, typed operations, runtime scale and grid jobs remain.
Optional compiler-bound native results are retained in Misc/NativeCorrect.lean. -/
namespace VeriTile.Bench.Repairs.LogsumexpFwd.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev original := @PairIO.original
abbrev candidate := @PairIO.candidate

/-- Exact complete row/block-grid equivalence on the original all-value input
scope. The explicit job permutation covers every scratch result and full frame. -/
specification source_equivalence {A : Type} [Inhabited A] (engines : Nat → Engine A)
    (a : RawKernel.Geometry) (scale : A) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (a.rows * a.blocks))) :
    original engines a scale jobs ≡[[]] candidate engines a scale jobs :=
  ProgramIO.Implementation.of_equivalent (PairIO.variant engines a scale jobs)
    (PairIO.equivalent engines a scale jobs schedule)

#axiomsClean source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd.FPEquiv
