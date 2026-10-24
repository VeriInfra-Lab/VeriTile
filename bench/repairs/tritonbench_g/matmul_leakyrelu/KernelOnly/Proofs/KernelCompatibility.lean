import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.KernelInterface
import bench.tritonbench_g.matmul_leakyrelu.Proofs.RealModel

/-! Original kernel endpoints retained unchanged for the Tensor applications. -/
namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section

theorem mathematical_kernel_raw
    (p : KernelInterface.Parameters) (initial : State ℝ)
    (A : Fin p.rows → Fin p.inner → ℝ) (B : Fin p.inner → Fin p.columns → ℝ)
    (hA : ∀ i k,Whole.inputA p.request initial i.val k.val=A i k)
    (hB : ∀ k j,Whole.inputB p.request initial k.val j.val=B k j)
    (separateA : p.C≠p.A) (separateB : p.C≠p.B) (schedule : Schedule p.request) :
    ∃ final, run (Whole.idealAlgebra id) Whole.Contracted.idealDotAcc p.request schedule.order initial=some final ∧
      (∀ i : Fin p.rows, ∀ j : Fin p.columns,
        final.mem p.C (Whole.outputOffset p.request i.val j.val)=Cell.mk .fp16 (Whole.mathematical p.request A B i j)) ∧
      (∀ region offset,¬Whole.Footprint p.request region offset →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,ran,result⟩ := scheduled_grid (Whole.idealAlgebra id) Whole.Contracted.idealDotAcc (Whole.Contracted.idealNumerics id) (Whole.Contracted.ideal_primitives id) p.request schedule initial separateA separateB
  refine ⟨final,ran,?_,result.2⟩
  intro i j
  have value := result.1 i.val i.isLt j.val j.isLt
  rw [Whole.Contracted.ideal_reference id p.request initial A B hA hB i j] at value
  exact value

#axiomsClean mathematical_kernel_raw
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
abbrev Word := VeriTile.Torch.FP32.Word
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem floating_kernel_raw (M : Algebra Word) (D : FPFused.DotAcc Word)
    (Q : Whole.Contracted.Numerics Word) (law : Whole.Contracted.PrimitiveLaws M D Q)
    (p : KernelInterface.Parameters) (initial : State Word)
    (A : Fin p.rows → Fin p.inner → Word) (B : Fin p.inner → Fin p.columns → Word)
    (hA : ∀ i k,Whole.inputA p.request initial i.val k.val=A i k)
    (hB : ∀ k j,Whole.inputB p.request initial k.val j.val=B k j)
    (separateA : p.C≠p.A) (separateB : p.C≠p.B) (schedule : Schedule p.request) :
    ∃ final, run M D p.request schedule.order initial=some final ∧
      (∀ i : Fin p.rows, ∀ j : Fin p.columns,
        final.mem p.C (Whole.outputOffset p.request i.val j.val)=Cell.mk .fp16 (Whole.Contracted.logicalReference Q p.request A B i j)) ∧
      (∀ region offset,¬Whole.Footprint p.request region offset →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,ran,result⟩ := scheduled_grid M D Q law p.request schedule initial separateA separateB
  refine ⟨final,ran,?_,result.2⟩
  intro i j
  have value := result.1 i.val i.isLt j.val j.isLt
  rw [Whole.Contracted.reference_logical Q p.request initial A B hA hB i j] at value
  exact value

#axiomsClean floating_kernel_raw
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
