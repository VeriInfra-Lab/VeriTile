import bench.tritonbench_g.matmul_triton2.Reference
import bench.tritonbench_g.matmul_triton2.Proofs.Real
import bench.tritonbench_g.matmul_triton2.Proofs.Grid

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.KernelInputs
open VeriTile.Triton FP.Structural Revision001
noncomputable section
variable {α : Type} [Inhabited α]

theorem real_reference (p : Core.Parameters) (initial : State ℝ)
    (a b : Nat → Nat → ℝ)
    (ha : ∀ r k, r<p.rows → k<p.innerSize →
      (initial.mem p.input (r*p.inputRow+k*p.inputInner)).read .real=a r k)
    (hb : ∀ k c, k<p.innerSize → c<p.columns →
      (initial.mem p.weight (k*p.weightInner+c*p.weightColumn)).read .real=b k c)
    (r c : Nat) (hr : r<p.rows) (hc : c<p.columns) (hBK : 0<p.blockInner) :
    Core.logical Core.realPrimitives p initial r c=Reference.real p.innerSize a b r c := by
  rw [Core.logical_realMatmul p initial r c hBK]
  unfold Core.realMatmul Reference.real
  apply RealSum.sum_congr
  intro k hk
  simp only [Core.realProduct, ha r k hr hk, hb k c hk hc]

#specNonCircular Reference.real avoiding [Core.source,FP.Structural.exec]
#axiomsClean real_reference
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.KernelInputs
