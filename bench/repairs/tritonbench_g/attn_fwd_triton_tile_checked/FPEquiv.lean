import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.PairIO

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Equivalence
open VeriTile Triton FP.Structural AttnFwdTriton.TypedExecution Graph ReadLocality
open scoped VeriTile.Spec
noncomputable section

abbrev original (p : Config) :=
  ProgramIO.Implementation.original (NativeInterface.programIO p) (fun _ => True)
abbrev quotientRemainder (p : Config) :=
  ProgramIO.Implementation.ofVariant (PairIO.candidate p) (fun _ => True)

/-- Complete actual source comparison with the same TF32/FP16 graph and BF16
store, all prior compound logical inputs, successful safe runs and frames. -/
specification quotient_remainder_equiv (p : Config) (stage : p.stage=3)
    (nonoverlap : NonOverlapping p) (separate : Separated p) :
    original p ≡[[]] quotientRemainder p := by
  exact ProgramIO.Implementation.of_equivalent (PairIO.candidate p)
    (PairIO.equivalent p stage nonoverlap separate)

#axiomsClean quotient_remainder_equiv
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Equivalence
