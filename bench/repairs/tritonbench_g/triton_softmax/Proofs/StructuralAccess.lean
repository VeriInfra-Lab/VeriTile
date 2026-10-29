import bench.repairs.tritonbench_g.triton_softmax.Kernels
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.TritonSoftmax.StructuralAccess
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Both precision profiles share exactly these actual masked source addresses. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0<B) (bounds : RegionName → Nat) (s : State α)
    (hin : ∀ i : Fin B,i.val<n → ib+s.pids 0*ir+i.val*ic<bounds inp)
    (hout : ∀ i : Fin B,i.val<n → ob+s.pids 0*orow+i.val*oc<bounds out) :
    FP.Structural.TraceSafe M bounds (softmax_kernel out inp ib ob ir ic orow oc n B) s := by
  obtain ⟨k,rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  simp [FP.Structural.TraceSafe,softmax_kernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    store,step,evalExpr,evalComputeOp,evalOp_unfold,State.setReg,numeric,natLt,bop,
    ComputeDType.eraseDType,TileShape.axisDim,Option.bind,Function.comp_def]
  exact ⟨fun i hi => hin i hi,fun i hi => hout i hi⟩

/-- Both precision profiles share exactly these actual masked source addresses. -/
theorem candidate_safe {α : Type} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0<B) (bounds : RegionName → Nat) (s : State α)
    (hin : ∀ i : Fin B,i.val<n → ib+s.pids 0*ir+i.val*ic<bounds inp)
    (hout : ∀ i : Fin B,i.val<n → ob+s.pids 0*orow+i.val*oc<bounds out) :
    FP.Structural.TraceSafe M bounds (indexCommutedKernel out inp ib ob ir ic orow oc n B) s := by
  obtain ⟨k,rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  simp [FP.Structural.TraceSafe,indexCommutedKernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    store,step,evalExpr,evalComputeOp,evalOp_unfold,State.setReg,numeric,natLt,bop,
    ComputeDType.eraseDType,TileShape.axisDim,Option.bind,Function.comp_def]
  exact ⟨fun i hi => by simpa [Nat.mul_comm] using hin i hi,fun i hi => hout i hi⟩

end
end VeriTile.Bench.Repairs.TritonSoftmax.StructuralAccess
