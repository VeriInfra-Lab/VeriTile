import bench.tritonbench_g.matmul_tma.Kernels
import bench.tritonbench_g.matmul_tma.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.RealExecution
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section

/-- Independent matrix product of the raw FP16 source-memory observations. -/
def matrixProduct (h : Shape) (f : Flags) (s : State ℝ) (i : CIndex (params h f)) : ℝ :=
  ∑ k : Fin h.inner, aValues (params h f) s (i.1,k,PUnit.unit) *
    bValues (params h f) s (k,i.2.1,PUnit.unit)

/-- Complete shared quotation, all matrix-product outputs and the exterior cell frame. -/
theorem kernel_math (rest : Algebra ℝ) (h : Shape) (f : Flags) (s : State ℝ) :
    VeriTile.Spec.Real (∃ t,
      exec (backend (StaticChecked.RealModel.algebra rest) StaticChecked.RealModel.dot) (kernel h f) s=some t ∧
      Post (params h f) f.half s t (matrixProduct h f s)) := by
  obtain ⟨t,execution,_,post⟩ := StaticChecked.executes (StaticChecked.RealModel.algebra rest)
    StaticChecked.RealModel.dot h f s
  have target : outputValues (StaticChecked.RealModel.algebra rest) StaticChecked.RealModel.dot
      (params h f) f.half s=matrixProduct h f s := by
    funext i
    simp only [outputValues,StaticChecked.RealModel.algebra,StaticChecked.RealModel.dot,
      StaticChecked.RealModel.contraction,matrixProduct]
    apply Finset.sum_congr rfl
    intro k _
    rfl
  rw [target] at post
  exact ⟨t,execution,post⟩

#specNonCircular matrixProduct avoiding [kernel,source,FP.Structural.exec,scatter]
#axiomsClean kernel_math
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.RealExecution
