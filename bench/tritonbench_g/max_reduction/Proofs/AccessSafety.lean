import bench.tritonbench_g.max_reduction.IO
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MaxReduction.AccessSafety
open VeriTile.Triton FP.Structural PublicIO
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem stage (e : Source.Engine α) (second candidate : Bool) (inp out : RegionName)
    (N B : Nat) (hb : 0<B) (s : State α) (bounds : RegionName→Nat)
    (hi : ∀ j : Fin B,stageStart second B (s.pids 0)+j.val<N →
      stageStart second B (s.pids 0)+j.val<bounds inp)
    (ho : stageOutput second (s.pids 0)<bounds out) :
    TraceSafe e.algebra bounds (stageSource second candidate inp out N B) s := by
  cases second <;> cases candidate <;>
    simp [stageSource,stage1, VeriTile.Bench.Repairs.MaxReduction.stage1, VeriTile.Bench.Repairs.MaxReduction.stage1Commuted, VeriTile.Bench.Repairs.MaxReduction.stage2, VeriTile.Bench.Repairs.MaxReduction.stage2Complement, VeriTile.Bench.Repairs.MaxReduction.dimension, VeriTile.Bench.Repairs.MaxReduction.dimensionCommuted, stage1Commuted,stage2, stage2Complement, TraceSafe,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
      ComputeDType.eraseDType,numeric,bop,natLt,natLe,numericLt,numericLe,Region.cast,
      Source.Engine.algebra,Source.reduceWith,TileShape.axisDim,hb,State.setReg,
      Nat.mul_comm,Nat.not_le,stageStart,stageOutput] at *
  all_goals exact ⟨hi,ho⟩

theorem dimension (e : Source.Engine α) (candidate : Bool) (inp ov oi : RegionName)
    (M N K BM BN : Nat) (hn : 0<BN) (s : State α) (bounds : RegionName→Nat)
    (hi : ∀ i : Fin BM,∀ j : Fin BN,s.pids 0*BM+i.val<M → j.val<N →
      (s.pids 0*BM+i.val)*N*K+j.val*K+s.pids 1<bounds inp)
    (hv : ∀ i : Fin BM,s.pids 0*BM+i.val<M → (s.pids 0*BM+i.val)*K+s.pids 1<bounds ov)
    (hx : ∀ i : Fin BM,s.pids 0*BM+i.val<M → (s.pids 0*BM+i.val)*K+s.pids 1<bounds oi) :
    TraceSafe e.algebra bounds
      (if candidate then dimensionCommuted inp ov oi M N K BM BN else MaxReduction.dimension inp ov oi M N K BM BN) s := by
  cases candidate <;>
    simp [MaxReduction.dimension,dimensionCommuted, VeriTile.Bench.Repairs.MaxReduction.stage1, VeriTile.Bench.Repairs.MaxReduction.stage1Commuted, VeriTile.Bench.Repairs.MaxReduction.stage2, VeriTile.Bench.Repairs.MaxReduction.stage2Complement, VeriTile.Bench.Repairs.MaxReduction.dimension, VeriTile.Bench.Repairs.MaxReduction.dimensionCommuted, TraceSafe,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
      ComputeDType.eraseDType,numeric,bop,natLt,natLe,numericLt,numericLe,Region.cast,
      Source.Engine.algebra,Source.reduceWith,Source.indexedWith,TileShape.axisDim,
      TileShape.insertAxisIndex,toFloat,ofFloat,hn,Nat.ne_of_gt hn,State.setReg,Nat.mul_comm] at *
  all_goals exact ⟨hi,hv,hx⟩

#axiomsClean stage
#axiomsClean dimension
end
end VeriTile.Bench.Repairs.MaxReduction.AccessSafety
