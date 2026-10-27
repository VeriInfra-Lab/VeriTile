import bench.repairs.tritonbench_g.softmax_triton3.Proofs.FP
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Floating
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- Actual structural accesses of the whole selected row source. This checks
all evaluated load/store addresses; no floating identity or result is assumed. -/
theorem source_trace_safe {α : Type} [Inhabited α] (M : Algebra α)
    (out inp mask : RegionName) (n B : Nat) (hasMask : Bool)
    (positive : 0<B) (bounds : RegionName → Nat) (s : State α)
    (input : ∀ j : Fin B,j.val<n → s.pids 0*n+j.val<bounds inp)
    (masks : hasMask=Bool.true → ∀ j : Fin B,j.val<n → s.pids 0*n+j.val<bounds mask)
    (output : ∀ j : Fin B,j.val<n → s.pids 0*n+j.val<bounds out) :
    TraceSafe M bounds (softmax_kernel out inp mask n n B hasMask) s := by
  cases hasMask <;>
    simp [← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add, TraceSafe,softmax_kernel,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp,store,State.setReg,numeric,numericLt,natLt,bop,
      TileShape.axisDim,positive,Option.bind,Function.comp_def,ComputeDType.eraseDType,toFloat,ofFloat]
  · exact ⟨input,output⟩
  · exact ⟨input,masks rfl,output⟩

#axiomsClean source_trace_safe
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.Floating
