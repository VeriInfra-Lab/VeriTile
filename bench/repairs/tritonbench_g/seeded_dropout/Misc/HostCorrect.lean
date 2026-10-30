import bench.repairs.tritonbench_g.seeded_dropout.Proofs.Memory

namespace VeriTile.Bench.Repairs.SeededDropout
open VeriTile.Triton VeriTile.Torch
open TritonBenchG.SeededDropout.Whole
noncomputable section
/-- Complete guarded host outcomes: exact returned Tensor, every logical output,
input and preexisting allocation preservation, and exterior output frame. -/
specification seeded_dropout_checked_real (x : TensorMetadata) (p : ℝ) (seed : Int)
    (admitted : checkInputs x (.finite p) (.integer seed) = .ok ())
    (binding : MetadataBinding x input) (rt : Runtime ℝ input entry preexisting)
    (api : ∀ a, rt.allocate = .ok a → AllocationAPI a) :
    ∃ outcome, runChecked realAlgebra x p seed rt = .ok outcome ∧
      HostSpec (realFormula p seed) rt outcome :=
  checked_real x p seed ((checkInputs_iff _ _ _).mp admitted) binding rt api
#axiomsClean seeded_dropout_checked_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SeededDropout
