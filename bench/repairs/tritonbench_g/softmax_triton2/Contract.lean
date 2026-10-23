import bench.repairs.tritonbench_g.softmax_triton2.Kernels
import bench.repairs.tritonbench_g.softmax_triton1.Contract
import Mathlib.Analysis.SpecialFunctions.Exp

namespace VeriTile.Bench.Repairs.SoftmaxTriton2
abbrev InputContract := VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous.InputContract
abbrev InputError := VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous.InputError
abbrev checkInput := VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous.checkInput
abbrev checkTensorInput := VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous.checkTensorInput
abbrev AdmittedArgument := VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous.AdmittedArgument

/-- Independent mathematical target over exactly the logical columns.
No block width, reduction order, native graph or source evaluator occurs here. -/
noncomputable def reference (n : Nat) (xs : Nat → ℝ) (c : Nat) : ℝ :=
  Real.exp (xs c) / ∑ j ∈ Finset.range n, Real.exp (xs j)

end VeriTile.Bench.Repairs.SoftmaxTriton2
