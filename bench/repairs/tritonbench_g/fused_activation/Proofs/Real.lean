import bench.repairs.tritonbench_g.fused_activation.Proofs.Source

namespace VeriTile.Bench.Repairs.FusedActivation.RealProof
open VeriTile.Triton FP.Structural
noncomputable section

def algebra (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with literal := fun _ _ value => value
              fp32Load := id
              binary := fun precision dtype operation x y =>
                match operation with
                | .add => x+y
                | .mul => x*y
                | .max => max x y
                | other => rest.binary precision dtype other x y
              unary := fun precision operation x =>
                match operation with
                | .sigmoid => Real.sigmoid x
                | other => rest.unary precision other x }

theorem scalar_real (rest : Algebra ℝ) (multiplier x bias input : ℝ) :
    scalar (algebra rest) multiplier Bool.true x bias input = Real.sigmoid (multiplier*input+x+bias) := rfl

theorem scalar_branch (rest : Algebra ℝ) (activation : Bool) (multiplier x bias input : ℝ) :
    scalar (algebra rest) multiplier activation x bias input =
      if activation then Real.sigmoid (multiplier*input+x+bias) else max 0 (multiplier*input+x+bias) := by
  cases activation <;> rfl

#axiomsClean scalar_real
end
end VeriTile.Bench.Repairs.FusedActivation.RealProof
