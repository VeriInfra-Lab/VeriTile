import bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.Host

namespace VeriTile.Bench.Repairs.ReluTritonKernel.RealProofs
open VeriTile.Triton FP.Structural
noncomputable section

/-- Mathematical reference, independent of execution and pointer addressing. -/
def reference (x : ℝ) : ℝ := max 0 x

def backend (rest : Algebra ℝ) : Algebra ℝ :=
  {rest with literal:=fun _ _ x=>x
             fp32Load:=id
             compareLe:=fun p d=>if p=some .fp32 ∧ d=.real then
               some (fun x y=>decide (x≤y)) else rest.compareLe p d}

theorem compare (rest : Algebra ℝ) :
    (backend rest).compareLe (some .fp32) .real=some (fun x y=>decide (x≤y)) := by simp [backend]

theorem scalar (rest : Algebra ℝ) (x : ℝ) :
    Structural.scalar (backend rest) (fun x y=>decide (x≤y)) x=reference x := by
  simp only [Structural.scalar,backend,reference]
  by_cases h : 0≤x
  · simp [h,max_eq_right h]
  · simp [h,max_eq_left (le_of_not_ge h)]

#specNonCircular reference avoiding [kernel, Coordinates.kernel,Structural.scalar,Host.runWrapper]
#axiomsClean scalar
end
end VeriTile.Bench.Repairs.ReluTritonKernel.RealProofs
