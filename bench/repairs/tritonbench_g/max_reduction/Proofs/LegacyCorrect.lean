import bench.repairs.tritonbench_g.max_reduction.Proofs.Mathematical

namespace VeriTile.Bench.Repairs.MaxReduction.Correct
open Execution VeriTile.Torch
noncomputable section

theorem max_real (x : TensorMetadata) (h : GlobalContract x) (profile : Host.GlobalProfile x.numel)
    (input : Nat → ℝ) (s : Heap (WithBot ℝ))
    (binding : ∀ i,i<x.numel → Execution.read ⊥ s "input" i=(input i : WithBot ℝ)) :
    ∃ (answer : ℝ) (result : Host.ScalarResult (WithBot ℝ)),
      Host.max Compute.real x profile s=.ok result ∧
      result.shape=[] ∧ result.dtype=.floating .fp32 ∧ result.device=x.device ∧
      result.heap ("output",0)=.value (answer : WithBot ℝ) ∧
      RealModel.IsMaximum input x.numel answer ∧
      (∀ q,(q.1≠"middle" ∨ globalMid x.numel≤q.2) → q≠("output",0) → result.heap q=s q) := by
  exact Mathematical.max_real x h profile input s binding

#axiomsClean max_real

theorem max_dim_real (a : DimArguments) (h : DimContract a)
    (trials : List (Host.DimProfile a.N)) (selected : Host.DimProfile a.N)
    (input : Nat → ℝ) (s : Heap (WithBot ℝ))
    (binding : ∀ i,i<a.input.numel → Execution.read ⊥ s "input" i=(input i : WithBot ℝ)) :
    ∃ result : Host.DimensionResult (WithBot ℝ),Host.maxDim Compute.real a trials selected s=.ok result ∧
      result.shape=a.outputShape ∧ result.valuesDtype=.floating .fp32 ∧
      result.indicesDtype=.other "int64" ∧ result.device=a.input.device ∧
      (∀ j,j<a.M*a.K → ∃ (answer : ℝ) (index : Nat),
        result.heap ("values",j)=.value (answer : WithBot ℝ) ∧
        result.heap ("indices",j)=.index (BitVec.ofNat 64 index) ∧
        RealModel.IsLeastArgmax (fun n => input (j/a.K*a.N*a.K+n*a.K+j%a.K)) a.N answer index) ∧
      (∀ q,(q.1≠"values" ∧ q.1≠"indices" ∨ a.M*a.K≤q.2) → result.heap q=s q) := by
  exact Mathematical.max_dim_real a h trials selected input s binding

#axiomsClean max_dim_real

end
end VeriTile.Bench.Repairs.MaxReduction.Correct
