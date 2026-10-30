import bench.tritonbench_g.rope_backward_transform.Proofs.ProgramPair

/-! Actual complete-program equivalence for the index common-subexpression
elimination candidate. Source FP32 arithmetic, casts and the chosen FMA lowering
are retained. The complete Q/K result reads include unchanged odd tails. -/
namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Original and optimized complete backward grids, under the retained native
FP32 backend and all original Tagged payloads, including nonfinite words.
No numerical transformation rule or additional input restriction is required. -/
specification source_candidate_equivalence (cfg : FP.Config) (r : Request) :
    PairIO.original (Numerics.operations (KernelContract.nativePrimitives cfg))
      (fun x => Numerics.leaf x.word) r ≡[[]]
    PairIO.indexShared (Numerics.operations (KernelContract.nativePrimitives cfg))
      (fun x => Numerics.leaf x.word) r := by
  exact ReadProgramIO.Implementation.of_equivalent (PairIO.candidate _ _ r)
    (PairIO.equivalent _ _ r)

#axiomsClean source_candidate_equivalence
end
end VeriTile.Bench.Repairs.RopeBackwardTransform
