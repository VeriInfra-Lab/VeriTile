import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Kernels
import bench.tritonbench_g.matmul_tma.Proofs.FPContract
namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourceExecution
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Registers are replaced by name, including when a cast changes carrier. -/
theorem overwriteReg {α : Type} (s : State α) (name : RegName)
    (d₁ d₂ : TileDType) (sh₁ sh₂ : TileShape)
    (v₁ : TileIndex sh₁ → FP.Structural.Value α d₁) (v₂ : TileIndex sh₂ → FP.Structural.Value α d₂) :
    (s.setReg name d₁ sh₁ v₁).setReg name d₂ sh₂ v₂ = s.setReg name d₂ sh₂ v₂ := by
  cases s
  simp only [State.setReg, State.mk.injEq, true_and, and_true]
  funext d sh n
  by_cases h : n = name <;> simp [h]

/-- Independent ordered dot, the explicit conditional conversion, and the
destination storage conversion. The two same-format storage conversions reduce
to identity in the selected real and native models, not in an arbitrary algebra. -/
def values {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (half : Bool) (s : State α) (i : CIndex p) : α :=
  let value := D.apply none {} (batch := []) (aValues p s) (bValues p s) i
  if half then M.cast none .fp16 .fp16 (M.cast none .fp32 .fp16 value)
  else M.cast none .fp32 .fp32 value

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (half : Bool) (s : State α) :
    ∃ t, exec (backend M D) (source p half) s = some t ∧
      t.mem = (scatter p half s (values M D p half s)).mem := by
  cases half <;>
    simp [source, FP.Structural.exec, ComputeKernel.surfaceBody, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
      numeric, bop, Region.cast, BlockPtr.inBounds, BlockPtr.address_2d_zero_offsets,
      TileShape.indexToList, backend, values, aValues, bValues,
      aOffset, bOffset, cOffset, outputDType, scatter, ofFloat, toFloat]
  · exact scatter_memory_congr (half := Bool.false) _ _ _ _ rfl
  · exact scatter_memory_congr (half := Bool.true) _ _ _ _ rfl

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (half : Bool) (s : State α) (hi : Function.Injective (cOffset p)) :
    ∃ t, exec (backend M D) (source p half) s = some t ∧
      Post p half s t (values M D p half s) := by
  obtain ⟨t,ran,mem⟩ := execution_memory M D p half s
  exact ⟨t,ran,post_of_memory p half s t _ hi mem⟩

#axiomsClean execution_memory
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourceExecution
