import bench.repairs.tritonbench_g.seeded_dropout.Proofs.Memory

namespace VeriTile.Bench.Repairs.SeededDropout
open VeriTile.Triton VeriTile.Torch
open TritonBenchG.SeededDropout.Whole
noncomputable section
/-- FP32 pointwise source formula includes canonical RNG, strict ordered threshold,
eager denominator/division, load and store. Primitive numerical conformance is
explicitly deferred; addressing, guards, masks and returned values are proved. -/
specification seeded_dropout_checked_fp32 (M : FP.Structural.Algebra (BitVec 32))
    (lt : BitVec 32 → BitVec 32 → Bool)
    (compare : M.compareLt (some .fp32) .real = some lt)
    (x : TensorMetadata) (p : ℝ) (seed : Int)
    (admitted : checkInputs x (.finite p) (.integer seed) = .ok ())
    (binding : MetadataBinding x input) (rt : Runtime (BitVec 32) input entry preexisting)
    (api : ∀ a, rt.allocate = .ok a → AllocationAPI a)
    (physical : ∀ a ha, PhysicalAPI x a (api a ha)) :
    (∃ outcome, runChecked (FP.ProfiledRewrite.engine M) x p seed rt = .ok outcome ∧
      HostSpec (ActualTile.scalar (FP.ProfiledRewrite.engine M) lt p seed) rt outcome) ∧
    ∀ a ha, ActiveAccess x a (api a ha) (physical a ha) := by
  have h := (checkInputs_iff _ _ _).mp admitted
  exact ⟨checked_fp32 M lt compare x p seed h binding rt api,
    fun a ha => active_access x p seed h binding a (api a ha) (physical a ha)⟩
#axiomsClean seeded_dropout_checked_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SeededDropout
