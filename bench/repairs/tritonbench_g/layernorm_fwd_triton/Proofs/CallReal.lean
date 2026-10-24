import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SuccessfulCall
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Real

/-! Independent arbitrary-Real interpretation of the same successful call.
Tensor handles, metadata, selected code, grid order and local event transports
are exactly those of `c`. There is no second successful call premise and no
FP-word decoding in the mathematical inputs or epsilon. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Wrapper
open Torch Address Memory Execution HostPrepared
noncomputable section
open scoped Classical
variable {α Store Byte Host : Type} {api : API α Store Byte} {neg : α → α} {caller : Nat → Prop}
  {platform : Runtime.Platform α} {service : Runtime.Service platform api neg caller Host}
  {lib : APIHost api neg caller Host} {q : Request} {initial : State Store Host}

def SuccessfulCall.realX (c : SuccessfulCall platform api neg service lib q initial)
    (m : Nat → ℝ) (row : Row c.setup.shape) (col : Nat) : ℝ :=
  if h : col<c.setup.shape.columns then realLogical m q.x (HostLayout.index3 row ⟨col,h⟩) else 0
def SuccessfulCall.realW (c : SuccessfulCall platform api neg service lib q initial)
    (m : Nat → ℝ) (row : Row c.setup.shape) (col : Nat) : ℝ :=
  if h : col<c.setup.shape.columns then realLogical m q.w (HostLayout.index2 row ⟨col,h⟩) else 0

def SuccessfulCall.realKernel (c : SuccessfulCall platform api neg service lib q initial)
    (m : Nat → ℝ) (eps : ℝ) : Nat → ℝ :=
  CompilerControl.execute RealProof.ops c.dispatch.code realAPI c.setup.prepared.layout eps
    c.dispatch.schedule.rows (c.preShadow m)
def SuccessfulCall.realFinal (c : SuccessfulCall platform api neg service lib q initial)
    (m : Nat → ℝ) (eps : ℝ) : Nat → ℝ :=
  ((lib.leave initial.owner).shadow c.kernelState).run (c.realKernel m eps)

theorem call_real_snapshots (c : SuccessfulCall platform api neg service lib q initial)
    (m : Nat → ℝ) (row : Row c.setup.shape) :
    xSnapshot RealProof.ops realAPI c.setup.prepared.layout (c.preShadow m) row=c.realX m row ∧
    wSnapshot RealProof.ops realAPI c.setup.prepared.layout (c.preShadow m) row=c.realW m row := by
  constructor <;> funext col <;> by_cases h : col<c.setup.shape.columns
  · simp only [xSnapshot,SuccessfulCall.realX,dif_pos h,HostPrepared.x_address,realAPI]
    have hc : Torch.Coordinate q.x (HostLayout.index3 row ⟨col,h⟩) := by
      simpa only [Torch.Coordinate,(call_inputs c).1.shape] using
        HostLayout.coordinate3 c.setup.prepared.xv row ⟨col,h⟩
    have hv := (call_inputs c).1.realValues m _ hc
    simpa only [realLogical,c.setup.prepared.xPositive,Bool.false_eq_true,if_false] using hv
  · simp only [xSnapshot,SuccessfulCall.realX,dif_neg h,RealProof.ops]
  · simp only [wSnapshot,SuccessfulCall.realW,dif_pos h,HostPrepared.w_address,realAPI]
    have hc : Torch.Coordinate q.w (HostLayout.index2 row ⟨col,h⟩) := by
      simpa only [Torch.Coordinate,(call_inputs c).2.shape] using
        HostLayout.coordinate2 c.setup.prepared.wv row ⟨col,h⟩
    have hv := (call_inputs c).2.realValues m _ hc
    simpa only [realLogical,c.setup.prepared.wPositive,Bool.false_eq_true,if_false] using hv
  · simp only [wSnapshot,SuccessfulCall.realW,dif_neg h,RealProof.ops]

theorem call_real_readback (c : SuccessfulCall platform api neg service lib q initial)
    (m : Nat → ℝ) (eps : ℝ) (row : Row c.setup.shape) (col : Fin c.setup.shape.columns) :
    realLogical (c.realFinal m eps) c.returned (HostLayout.index3 row col)=
      RealProof.outputSpec c.setup.shape.columns (c.realX m row) (c.realW m row) eps col.val := by
  have hv := ((lib.leave initial.owner).shadow c.kernelState).values (c.realKernel m eps) c.returned
    (call_kernel_output_live c) _ (HostLayout.coordinate3 c.setup.prepared.yv row col)
  change realLogical (c.realFinal m eps) c.returned _ = realLogical (c.realKernel m eps) c.returned _ at hv
  rw [hv]
  change realLogical (c.realKernel m eps) c.setup.prepared.y _ = _
  simp only [realLogical,c.setup.prepared.yPositive,Bool.false_eq_true,if_false]
  unfold SuccessfulCall.realKernel
  rw [CompilerControl.execute_selected,Compiler.execute_grid,←HostPrepared.y_address]
  change realAPI.read (executeGrid RealProof.ops c.dispatch.code.plan realAPI c.setup.prepared.layout
    eps c.dispatch.schedule.rows (c.preShadow m)) (yAddress c.setup.prepared.layout row col)=_
  rw [grid_readback _ _ realAPI _ _ _ _ (c.dispatch.schedule.complete c.dispatch.completed),
    (call_real_snapshots c m row).1,(call_real_snapshots c m row).2]
  exact RealProof.reference_real ..

theorem call_real_frame (c : SuccessfulCall platform api neg service lib q initial)
    (m : Nat → ℝ) (eps : ℝ) (i : Nat) (hi : initial.backing i) : c.realFinal m eps i=m i := by
  have hc : caller i := by rw [c.setup.domain.backing];exact hi
  unfold SuccessfulCall.realFinal
  rw [((lib.leave initial.owner).shadow c.kernelState).frame _ i hc]
  unfold SuccessfulCall.realKernel
  rw [CompilerControl.execute_selected,Compiler.execute_grid]
  have hf := grid_frame RealProof.ops c.dispatch.code.plan realAPI c.setup.prepared.layout
    (c.preShadow m) eps c.dispatch.schedule.rows i (HostPrepared.caller_outside c.setup.prepared i hc)
  exact hf.trans (call_pre_shadow_frame c m i hc)

#axiomsClean call_real_snapshots
#axiomsClean call_real_readback
#axiomsClean call_real_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Wrapper
end VeriTile.Bench.Repairs.LayernormFwdTriton
