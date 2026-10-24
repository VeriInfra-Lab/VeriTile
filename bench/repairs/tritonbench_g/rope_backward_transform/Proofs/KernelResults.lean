import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.KernelResults
open VeriTile.Triton FP.Structural
open KernelContract (launchRows launched)
noncomputable section

/-- Complete raw Q/K source execution and fixed-coefficient input adjoint,
including unchanged odd tails and the exact all-region frame. -/
theorem rope_backward_kernel_correct (r : Request) (m : Source.Memory ℝ) :
    ∃ final, GridExec ideal.backend r (launchRows r) m final ∧
      (∀ key row head col, row<r.rows → head<r.heads key → col<r.dimension →
        (final (region key) (r.offset key row head col)).read .fp32 =
          KernelInterface.realValue r m key row head col) ∧
      (∀ name offset, ¬ KernelInterface.Written r name offset → final name offset=m name offset) := by
  obtain ⟨final, executed, values, frame⟩ := KernelInterface.complete ideal r m
  refine ⟨final, executed, ?_, frame⟩
  intro key row head col hr hh hc
  exact (values key row head col hr hh hc).trans (KernelInterface.real_binding r m key row head col)

theorem rope_backward_kernel_no_launch {α : Type} [Inhabited α] (M : Algebra α)
    (r : Request) (m : Source.Memory α)
    (off : r.rows=0 ∨ r.dimension<2 ∨ (r.qHeads=0 ∧ r.kHeads=0)) :
    launchRows r=[] ∧ GridExec M r (launchRows r) m m := by
  have noton : ¬launched r := by
    intro on
    rcases off with h | h | ⟨q,k⟩
    · have rows:=on.1; omega
    · have half:=on.2.1; unfold Request.half at half; omega
    · rcases on.2.2 with h | h <;> omega
  have empty : launchRows r=[] := by simp [launchRows,noton]
  exact ⟨empty,empty ▸ GridExec.nil m⟩

#axiomsClean rope_backward_kernel_correct
#axiomsClean rope_backward_kernel_no_launch
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.KernelResults


namespace VeriTile.Bench.Repairs.RopeBackwardTransform.KernelResults
open VeriTile.Triton FP.Structural
open KernelContract (launchRows nativePrimitives)
noncomputable section

/-- Complete raw typed source and its coefficient-first FP32 FMA profile,
all Q/K output observations and the exact regional frame. -/
theorem rope_backward_kernel_fp32 (cfg : FP.Config) (r : Request) (m : Source.Memory Numerics.Tagged) :
    ∃ final, GridExec (Numerics.operations (nativePrimitives cfg)).backend r (launchRows r) m final ∧
      (∀ key row head col, row<r.rows → head<r.heads key → col<r.dimension →
        ((final (region key) (r.offset key row head col)).read .fp32).word =
          Numerics.result (nativePrimitives cfg) r
            (fun key o => ((m (region key) o).read .fp32).word)
            (fun o => ((m "C" o).read .fp32).word)
            (fun o => ((m "S" o).read .fp32).word) key row head col) ∧
      (∀ name offset, ¬ KernelInterface.Written r name offset → final name offset=m name offset) := by
  obtain ⟨final, executed, values, frame⟩ := KernelInterface.complete (Numerics.operations (nativePrimitives cfg)) r m
  refine ⟨final, executed, ?_, frame⟩
  intro key row head col hr hh hc
  rw [values key row head col hr hh hc]
  exact Numerics.source_profile (nativePrimitives cfg) r m key row head col

#axiomsClean rope_backward_kernel_fp32
#specNonCircular Numerics.result avoiding [Source.kernel, grid, scalarResult, FP.Structural.exec]
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.KernelResults
