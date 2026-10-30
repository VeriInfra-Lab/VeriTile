import bench.repairs.tritonbench_g.triton_matmul.Proofs.Native
import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Kernels
import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Raw
import bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelRealModel
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.Operational
open VeriTile.Triton FP.Structural FPValue KernelRealModel
noncomputable section

/-- Independent full-K GEMM over the initial raw input cells. -/
def mathematical (g : Geometry) (before : State ℝ) (i : Fin g.rows) (j : Fin g.columns) : ℝ :=
  SourceScope.mathematical g.rows g.columns g.inner
    (fun i k => inputA g.request before i.val k.val)
    (fun k j => inputB g.request before k.val j.val) i j

/-- The whole shared-source grid succeeds in every complete scheduling order,
with independent GEMM outputs and the exact exterior typed-memory frame. -/
theorem kernel_real (g : Geometry) (before : State ℝ) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C≠g.A) (sepB : g.C≠g.B) :
    VeriTile.Spec.Real (∃ final,runPrograms ideal idealDot g.request ids before=some final ∧
      (∀ i : Fin g.rows,∀ j : Fin g.columns,
        final.mem g.C (outputOffset g.request i.val j.val)=stored g.storage (mathematical g before i j)) ∧
      (∀ region offset,¬Footprint g.request region offset → final.mem region offset=before.mem region offset)) := by
  obtain ⟨final,ran,values,frame⟩ := grid_order_correct ideal idealDot idealNumerics g.request
    (ideal_primitives g.storage) before ids order sepA sepB
  refine ⟨final,ran,?_,frame⟩
  intro i j
  have value := values i.val i.isLt j.val j.isLt
  rw [ideal_reference g.request before
    (fun i k => inputA g.request before i.val k.val)
    (fun k j => inputB g.request before k.val j.val)
    (fun _ _ => rfl) (fun _ _ => rfl) i j] at value
  exact value

#specNonCircular mathematical avoiding [Source.typed,runPrograms,FP.Structural.exec]
#axiomsClean kernel_real
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.Operational

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.Operational
open VeriTile.Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Independent fixed-instruction recurrence over the initial raw cells. -/
def nativeExpected (R : Native.Reference) (g : Geometry) (before : State Word) (row col : Nat) : Word :=
  reference (Native.numerics (Native.referenceInstructions R) g.storage) g.request before row col

/-- Complete typed shared-source grid, all native storage words and full
exterior-cell frame. The four original fixed instruction/encoding laws remain explicit. -/
theorem kernel_native_if_local (rest : Algebra Word) (I : Native.Instructions)
    (R : Native.Reference) (numeric : Native.Correspondence I R)
    (g : Geometry) (before : State Word) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C≠g.A) (sepB : g.C≠g.B) :
    ∃ final,runPrograms (Native.algebra rest I) (Native.dotAcc I) g.request ids before=some final ∧
      (∀ row,row<g.rows → ∀ col,col<g.columns →
        final.mem g.C (outputOffset g.request row col)=stored g.storage (nativeExpected R g before row col)) ∧
      (∀ region offset,¬Footprint g.request region offset → final.mem region offset=before.mem region offset) := by
  obtain ⟨final,ran,values,frame⟩ := grid_order_correct (Native.algebra rest I) (Native.dotAcc I)
    (Native.numerics I g.storage) g.request (Native.primitives rest I g.storage) before ids order sepA sepB
  refine ⟨final,ran,?_,frame⟩
  intro row hr col hc
  simpa only [Native.instructions_eq I R numeric] using values row hr col hc

#specNonCircular nativeExpected avoiding [Source.typed,runPrograms,FP.Structural.exec]
#axiomsClean kernel_native_if_local
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.Operational
