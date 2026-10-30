import bench.tritonbench_g.swiglu_backward.Proofs.Math
import Mathlib.Analysis.SpecialFunctions.Sigmoid
namespace VeriTile.Bench.TritonBenchG.SwigluBackward.Math

/-- The gate partial derivative is derived from the independent forward
function, not introduced as a kernel oracle. -/
theorem forward_gate_derivative (x y : ℝ) :
    HasDerivAt (fun t => forward t y) (y*sigmoid x*(1+x*(1-sigmoid x))) x := by
  have h := ((hasDerivAt_id x).mul (Real.hasDerivAt_sigmoid x)).mul_const y
  convert h using 1
  · funext t; simp [forward,sigmoid_eq]
  · simp only [sigmoid_eq,id_eq]
    ring

/-- The value partial derivative. The public VJP multiplies both derivatives
by the supplied upstream cotangent. -/
theorem forward_value_derivative (x y : ℝ) :
    HasDerivAt (fun t => forward x t) (x*sigmoid x) y := by
  simpa [forward] using (hasDerivAt_id y).const_mul (x*sigmoid x)

theorem vjp (x y d : ℝ) :
    dx x y d = d*(y*sigmoid x*(1+x*(1-sigmoid x))) ∧
    dy x d = d*(x*sigmoid x) := by
  constructor <;> simp only [dx,dy] <;> ring

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluBackward.Math
