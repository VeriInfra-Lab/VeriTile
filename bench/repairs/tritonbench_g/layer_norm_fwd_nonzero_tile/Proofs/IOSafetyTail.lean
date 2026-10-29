import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.IOSafetyPrefix

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.IOSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
open TypedSource TypedSupport
noncomputable section
set_option maxHeartbeats 20000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem normalization_safe {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s : State α) (hs : Ready f r stride block row s)
    (bounds : RegionBounds) (access : FlatDomain.RowBounds f r stride M N row bounds) :
    Safety.traceSafeList (engine A) bounds (normalization f r stride M N eps p block hb) s := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
  cases hrms : f.rms
  all_goals simp (config := {maxSteps:=500000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    normalization,repaired,original,ComputeKernel.surfaceBody,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
    numericLt,Region.cast,Random.i32ArangeSize,TileShape.axisDim,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,Broadcast.leftIndex,Broadcast.rightIndex,hrms]


  all_goals repeat' apply And.intro
  all_goals first | exact access.mean hrms | exact access.rstd

theorem affine_safe {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s : State α) (hs : NormalizedReady f r stride block row s)
    (bounds : RegionBounds) (access : FlatDomain.RowBounds f r stride M N row bounds) :
    Safety.traceSafeList (engine A) bounds (affine f r stride M N eps p block hb) s := by
  rcases hs with ⟨⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩,⟨std,hstd⟩,hm⟩
  cases hrms : f.rms
  · obtain ⟨mean,hmean⟩ := hm hrms
    cases hbias : f.bias <;> cases hw1 : f.w1 <;> cases hb1 : f.b1
    all_goals simp (config := {maxSteps:=500000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    affine,repaired,original,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hmean,hrms,hbias,hw1,hb1]

    all_goals repeat' apply And.intro
    all_goals first
    | intro j on; exact access.weight j.val on
    | intro j on; exact access.bias hbias j.val on
    | intro j on; exact access.y j.val on
    | intro j on; exact access.weight1 hw1 j.val on
    | intro j on; exact access.bias1 hw1 hb1 j.val on
    | intro j on; exact access.y1 hw1 j.val on
  · cases hbias : f.bias <;> cases hw1 : f.w1 <;> cases hb1 : f.b1
    all_goals simp (config := {maxSteps:=500000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    affine,repaired,original,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hrms,hbias,hw1,hb1]

    all_goals repeat' apply And.intro
    all_goals first
    | intro j on; exact access.weight j.val on
    | intro j on; exact access.bias hbias j.val on
    | intro j on; exact access.y j.val on
    | intro j on; exact access.weight1 hw1 j.val on
    | intro j on; exact access.bias1 hw1 hb1 j.val on
    | intro j on; exact access.y1 hw1 j.val on

#axiomsClean normalization_safe
#axiomsClean affine_safe
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.IOSafety
