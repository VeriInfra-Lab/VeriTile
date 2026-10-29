import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.SafetyForwardInner

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.ForwardInner
open VeriTile Triton FP.Structural Safety
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop FPForwardInner
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems FullSource
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem initial_safety (A : Algebra α) (s : State α) (X Y : RegionName)
    (rows N B : Nat) (bounds : RegionName → Nat) :
    TraceSafe A bounds (Source.forward_inner Y X rows N B Bool.false) s ↔
      rangeSafe A bounds "start_n" 0 (previous N B) B (rangeBody (code Y X rows N B) 5)
        (initial A s X Y N B) ∧
      ∀ mid, range A "start_n" 0 (previous N B) B (rangeBody (code Y X rows N B) 5)
        (initial A s X Y N B)=some mid →
        traceSafeList A bounds ((code Y X rows N B).drop 6) mid := by
  simp [TraceSafe,Source.forward_inner,softmax_kernel_inner,code,branch,rangeBody,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,initial,literal,previous,integralFloorDiv]
  all_goals delta bop
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,FPConstants.decimal_zero]
  all_goals split <;> simp_all
  all_goals intros; split <;> trivial

theorem between_safety (A : Algebra α) (base mid : State α) (X Y : RegionName)
    (row rows N B : Nat) (v : Acc α B) (hs : Ready base X Y row N B v mid)
    (bounds : RegionName → Nat) :
    traceSafeList A bounds ((code Y X rows N B).drop 6) mid ↔
      rangeSafe A bounds "start_n" (previous N B) N B (rangeBody (code Y X rows N B) 6) mid ∧
      ∀ t, range A "start_n" (previous N B) N B (rangeBody (code Y X rows N B) 6) mid=some t →
        traceSafeList A bounds ((code Y X rows N B).drop 7) t := by
  rcases hs with ⟨hx,hy,hprev,hm,hz,hmem⟩
  simp [code,branch,rangeBody,Source.forward_inner,softmax_kernel_inner,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,hprev]
  all_goals split <;> simp_all

theorem after_safety (A : Algebra α) (base mid : State α) (X Y : RegionName)
    (row rows N B : Nat) (hB : 0<B) (v : Acc α B)
    (hs : Ready base X Y row N B v mid) (bounds : RegionName → Nat) :
    traceSafeList A bounds ((code Y X rows N B).drop 7) mid ↔
      rangeSafe A bounds "start_n" 0 B B (rangeBody (code Y X rows N B) 11)
        (writeInitial A mid N B v) ∧
      ∀ t, range A "start_n" 0 B B (rangeBody (code Y X rows N B) 11)
        (writeInitial A mid N B v)=some t →
        traceSafeList A bounds ((code Y X rows N B).drop 12) t := by
  rcases hs with ⟨hx,hy,hprev,hm,hz,hmem⟩
  simp [code,branch,rangeBody,Source.forward_inner,softmax_kernel_inner,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hx,hy,hprev,hm,hz,hmem,writeInitial,maxReduced,sumReduced,
    previous,integralFloorDiv,hB,FPReference.mul,FPReference.sub,FPReference.exp]
  all_goals delta bop maxReduced
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_some]
  all_goals try simp only [Option.bind,hx,hy,hm,hz,String.reduceEq,↓reduceIte,↓reduceDIte]
  all_goals
    have eta (i : TileIndex [B]) : (i.1,PUnit.unit)=i := by rcases i with ⟨a,u⟩; cases u; rfl
    have scalar_eta (f : Values α .real []) : (fun _ : TileIndex [] => f PUnit.unit)=f := by
      funext i; cases i; rfl
    simp only [Function.comp_apply,eta,scalar_eta]
  all_goals split <;> simp_all

theorem last_safety (A : Algebra α) (s : State α) (X Y : RegionName)
    (rows N B : Nat) (bounds : RegionName → Nat) :
    traceSafeList A bounds ((code Y X rows N B).drop 12) s ↔
      rangeSafe A bounds "start_n" B N B (rangeBody (code Y X rows N B) 12) s := by
  simp [code,branch,rangeBody,Source.forward_inner,softmax_kernel_inner,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold]
  all_goals split <;> simp

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.ForwardInner
