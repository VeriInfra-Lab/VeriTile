import bench.tritonbench_g.rotary_transform.Proofs.ProgramPair

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Actual complete FP32 grid pair for interleaved lane-index CSE. All original
flags, signed metadata widths, arbitrary native words and logical domain remain
unchanged. Both result windows and raw frames include the original suffix
behavior; safe successful execution of each source is proved. -/
specification source_candidate_equivalence (c : Config) (valid : c.Valid)
    (p : Native.Instructions) (cu64 off64 : Bool) :
    PairIO.original c p cu64 off64 ≡[[]] PairIO.indexShared c p cu64 off64 := by
  exact ReadProgramIO.Implementation.of_equivalentOn (PairIO.candidate _ _ c cu64 off64)
    (PairIO.equivalent c valid p cu64 off64)

#axiomsClean source_candidate_equivalence
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked
