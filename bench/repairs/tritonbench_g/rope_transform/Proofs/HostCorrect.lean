import bench.repairs.tritonbench_g.rope_transform.Proofs.Public

namespace VeriTile.Bench.Repairs.RopeTransform
/-- Checked full forward RoPE mathematical projection, including singleton
storage reuse and all four returned objects. No numerical equality is assumed. -/
theorem rope_forward_correct (r : Request) (call : Host.Preparation ℝ r)
    (accepted : Checked.check r call=Bool.true) : Checked.RealOutcome r call :=
  Checked.whole_real r call accepted
#axiomsClean rope_forward_correct
end VeriTile.Bench.Repairs.RopeTransform
