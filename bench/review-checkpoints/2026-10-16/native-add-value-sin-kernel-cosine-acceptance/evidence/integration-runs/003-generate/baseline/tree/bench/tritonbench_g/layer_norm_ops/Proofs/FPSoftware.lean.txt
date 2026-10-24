import bench.tritonbench_g.layer_norm_ops.Proofs.FPContract
import bench.tritonbench_g.token_attn_reduceV.Proofs.FPSoftware
import bench.tritonbench_g.layer_norm_liger.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages (literal32)
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed (sequentialSum)
noncomputable section

/-- ScalarOps at every used fp32 operation, identity carrier casts and explicit
ascending-lane +0-seeded reduction. Nontrivial N conversion is named, not
assumed equal to Real arithmetic. No transcendental callback is used here. -/
def software (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (literal : ℝ → FP.Value .fp32) : Algebra (FP.Value .fp32) :=
  { VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software rest cfg with
    literal := fun _ _ x => literal32 literal x
    reduceSum := fun _ {_} ax keep xs => sequentialSum cfg ax keep xs }

/-- The two accumulator additions remain concrete rounded operations even
when the initial accumulator is positive zero. This isolates the arithmetic
whose omission in the old Real helper changed signed-zero behavior. -/
theorem dw_db_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (p : Parameters)
    (s : State (FP.Value .fp32)) (plain : Bool) (i : TileIndex [p.block]) :
    stored (software rest cfg literal) (rowDW (software rest cfg literal) p s plain i)=
      .mk .fp32 (FP.add cfg (FP.Value.zero .fp32)
        (FP.mul cfg (dyValue (software rest cfg literal) p s i)
          (xhatValue (software rest cfg literal) p s plain i))) ∧
    stored (software rest cfg literal) (rowDB (software rest cfg literal) p s i)=
      .mk .fp32 (FP.add cfg (FP.Value.zero .fp32) (dyValue (software rest cfg literal) p s i)) := by
  simp only [stored, rowDW, rowDB, op, zero]
  simp [software, literal32, VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

#axiomsClean dw_db_software
end
end VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
