import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SuccessfulCall

/-! Ordered FP readback of the actual returned allocation, with source logical
inputs snapshotted before any host event. No output relation is assumed. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Wrapper
open Torch Address Memory Execution HostPrepared
noncomputable section
open scoped Classical
variable {α Store Byte Host : Type} {api : API α Store Byte} {neg : α → α} {caller : Nat → Prop}
  {platform : Runtime.Platform α} {service : Runtime.Service platform api neg caller Host}
  {lib : APIHost api neg caller Host} {q : Request} {initial : State Store Host}

def SuccessfulCall.fpX (c : SuccessfulCall platform api neg service lib q initial)
    (row : Row c.setup.shape) (col : Nat) : α :=
  if h : col<c.setup.shape.columns then
    logical api neg initial.memory q.x (HostLayout.index3 row ⟨col,h⟩)
  else (platform.scalar c.dispatch.code.object.primitives).zero
def SuccessfulCall.fpW (c : SuccessfulCall platform api neg service lib q initial)
    (row : Row c.setup.shape) (col : Nat) : α :=
  if h : col<c.setup.shape.columns then
    logical api neg initial.memory q.w (HostLayout.index2 row ⟨col,h⟩)
  else (platform.scalar c.dispatch.code.object.primitives).zero

theorem call_x_coordinate (c : SuccessfulCall platform api neg service lib q initial)
    (row : Row c.setup.shape) (col : Fin c.setup.shape.columns) :
    Torch.Coordinate q.x (HostLayout.index3 row col) := by
  have h := HostLayout.coordinate3 c.setup.prepared.xv row col
  simpa only [Torch.Coordinate,(call_inputs c).1.shape] using h

theorem call_w_coordinate (c : SuccessfulCall platform api neg service lib q initial)
    (row : Row c.setup.shape) (col : Fin c.setup.shape.columns) :
    Torch.Coordinate q.w (HostLayout.index2 row col) := by
  have h := HostLayout.coordinate2 c.setup.prepared.wv row col
  simpa only [Torch.Coordinate,(call_inputs c).2.shape] using h

theorem call_fp_snapshots (c : SuccessfulCall platform api neg service lib q initial)
    (row : Row c.setup.shape) :
    xSnapshot (platform.scalar c.dispatch.code.object.primitives) api c.setup.prepared.layout c.dispatch.s2.memory row=c.fpX row ∧
    wSnapshot (platform.scalar c.dispatch.code.object.primitives) api c.setup.prepared.layout c.dispatch.s2.memory row=c.fpW row := by
  constructor <;> funext col <;> by_cases h : col<c.setup.shape.columns
  · simp only [xSnapshot,SuccessfulCall.fpX,dif_pos h,HostPrepared.x_address]
    have hv := (call_inputs c).1.values _ (call_x_coordinate c row ⟨col,h⟩)
    simpa only [logical,c.setup.prepared.xPositive,Bool.false_eq_true,if_false] using hv
  · simp only [xSnapshot,SuccessfulCall.fpX,dif_neg h]
  · simp only [wSnapshot,SuccessfulCall.fpW,dif_pos h,HostPrepared.w_address]
    have hv := (call_inputs c).2.values _ (call_w_coordinate c row ⟨col,h⟩)
    simpa only [logical,c.setup.prepared.wPositive,Bool.false_eq_true,if_false] using hv
  · simp only [wSnapshot,SuccessfulCall.fpW,dif_neg h]

/-- The actual context exit preserves the already live returned allocation.
The reference includes every source pass, selected reduction tree, selected
FMA/separate carry, epsilon conversion and final FP32 cast. -/
theorem call_fp_readback (c : SuccessfulCall platform api neg service lib q initial)
    (row : Row c.setup.shape) (col : Fin c.setup.shape.columns) :
    logical api neg c.final.memory c.returned (HostLayout.index3 row col)=
      Native.reference (platform.scalar c.dispatch.code.object.primitives) c.dispatch.code.plan
        (c.fpX row) (c.fpW row) (platform.epsilon c.dispatch.code.object q.eps) col.val := by
  have hv := (lib.leave initial.owner).preserves c.kernelState |>.values c.returned
    (call_kernel_output_live c) _ (HostLayout.coordinate3 c.setup.prepared.yv row col)
  change logical api neg c.final.memory c.returned _ = logical api neg c.kernelState.memory c.returned _ at hv
  rw [hv]
  change logical api neg c.kernelState.memory c.setup.prepared.y _ = _
  simp only [logical,c.setup.prepared.yPositive,Bool.false_eq_true,if_false]
  unfold SuccessfulCall.kernelState dispatchedState
  rw [Runtime.successful_readback c.dispatch, (call_fp_snapshots c row).1,(call_fp_snapshots c row).2]

#axiomsClean call_fp_snapshots
#axiomsClean call_fp_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Wrapper
end VeriTile.Bench.Repairs.LayernormFwdTriton
