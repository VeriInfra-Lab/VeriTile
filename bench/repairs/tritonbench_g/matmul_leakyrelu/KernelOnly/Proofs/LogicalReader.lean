import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Proofs.OriginalReal
import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.IO
namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.ReaderProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
open ReaderIO
open scoped BigOperators LogicalProgramIO
noncomputable section

theorem target_eq (p : KernelInterface.Parameters) (order : Schedule p.request) : (view p order).lift (fun xs r c =>
      let value := ∑ k : Fin p.inner, xs.A r k * xs.B k c
      if p.activation == "leaky_relu" then
        if 0 ≤ value then value else (1 / 100 : ℝ) * value
      else value) = Interface.realReference p := by
  funext xs o j
  fin_cases o
  rfl

/-- Full judgment equivalence, retaining the original safety, schedule and frame. -/
theorem judgment_iff (p : KernelInterface.Parameters) (order : Schedule p.request) :
    (view p order ⊨ fun xs r c =>
      let value := ∑ k : Fin p.inner, xs.A r k * xs.B k c
      if p.activation == "leaky_relu" then
        if 0 ≤ value then value else (1 / 100 : ℝ) * value
      else value) ↔
      (Original.io (Whole.idealAlgebra id) Whole.idealDot p order).Implements (Interface.realReference p) := by
  rw [LogicalProgramIO.implements_iff, target_eq] <;> rfl

theorem correct (p : KernelInterface.Parameters)
    (separateA : p.C ≠ p.A) (separateB : p.C ≠ p.B) (order : Schedule p.request) :
    (view p order ⊨ fun xs r c =>
      let value := ∑ k : Fin p.inner, xs.A r k * xs.B k c
      if p.activation == "leaky_relu" then
        if 0 ≤ value then value else (1 / 100 : ℝ) * value
      else value) :=
  (judgment_iff p order).mpr (OriginalReal.real_implements p separateA separateB order)
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.ReaderProof
