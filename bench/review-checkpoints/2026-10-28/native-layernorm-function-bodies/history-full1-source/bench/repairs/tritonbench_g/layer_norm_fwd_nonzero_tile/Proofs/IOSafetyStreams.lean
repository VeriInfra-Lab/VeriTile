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

theorem firstStream_safe {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s : State α) (hs : Ready f r stride block row s)
    (bounds : RegionBounds) (access : FlatDomain.RowBounds f r stride M N row bounds) :
    Safety.traceSafeList (engine A) bounds (firstStream f r stride M N eps p block hb) s := by
  by_cases hN : N = 1
  all_goals
    rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
    have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
      (seed_valid ((s.mem r.seeds row).read .int) hb)
    have ho : ∀ j : Fin block, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
      fun j => hv.2 (j,PUnit.unit)
    have hu := hv.1
    simp at hu
    cases hsc : f.rowscale <;> cases hd : f.dropout <;> cases hm : f.dropoutMask
    all_goals simp (config := {maxSteps:=500000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      firstStream,actual,Source.kernel,Source.rawKernel,Source.rawKernelOne,hN,sourceFlags,sourceRegions,sourceStrides,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,integralMod,FP.Structural.bop,natLt,
      ←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
      numericLt,cmp.available,seed_valid,Region.cast,Random.i32ArangeSize,
      hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hu,ho,Broadcast.leftIndex,Broadcast.rightIndex,hsc,hd,hm]


    all_goals repeat' apply And.intro
    all_goals first
    | simpa [hN] using (access.rowscale hsc)
    | simpa [hN] using (access.seeds hd)
    | intro j on; simpa [hN] using (access.mask hd hm j.val (by simpa [hN] using on))

theorem secondInput_safe {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s : State α) (hs : Ready f r stride block row s)
    (bounds : RegionBounds) (access : FlatDomain.RowBounds f r stride M N row bounds) :
    Safety.traceSafeList (engine A) bounds (secondInput f r stride M N eps p block hb) s := by
  by_cases hN : N = 1
  all_goals
    rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
    have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
      (seed_valid ((s.mem r.seeds (M+row)).read .int) hb)
    have ho : ∀ j : Fin block, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
      fun j => hv.2 (j,PUnit.unit)
    have hu := hv.1
    simp at hu
    cases hxflag : f.x1 <;> cases hsc : f.rowscale <;> cases hd : f.dropout <;> cases hm : f.dropoutMask
    all_goals simp (config := {maxSteps:=500000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      secondInput,actual,Source.kernel,Source.rawKernel,Source.rawKernelOne,hN,sourceFlags,sourceRegions,sourceStrides,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,integralMod,FP.Structural.bop,natLt,
      ←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
      numericLt,cmp.available,seed_valid,Region.cast,Random.i32ArangeSize,
      hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hu,ho,Broadcast.leftIndex,Broadcast.rightIndex,hxflag,hsc,hd,hm]


    all_goals repeat' apply And.intro
    all_goals first
    | simpa [hN] using (access.rowscale1 hxflag hsc)
    | simpa [hN] using (access.seeds1 hxflag hd)
    | intro j on; simpa [hN] using (access.x1 hxflag j.val (by simpa [hN] using on))
    | intro j on; simpa [hN] using (access.mask1 hxflag hd hm j.val (by simpa [hN] using on))

theorem residual_safe {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s : State α) (hs : Ready f r stride block row s)
    (bounds : RegionBounds) (access : FlatDomain.RowBounds f r stride M N row bounds) :
    Safety.traceSafeList (engine A) bounds (TypedSource.residual f r stride M N eps p block hb) s := by
  by_cases hN : N = 1
  all_goals
    rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
    cases hrflag : f.residual <;> cases hoflag : f.residualOut
    all_goals simp (config := {maxSteps:=500000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      TypedSource.residual,actual,Source.kernel,Source.rawKernel,Source.rawKernelOne,hN,sourceFlags,sourceRegions,sourceStrides,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      ←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
      numericLt,Region.cast,Random.i32ArangeSize,
      hrow,hcols,hy,hx1,hres,hresout,hy1,hx,Broadcast.leftIndex,Broadcast.rightIndex,hrflag,hoflag]


    all_goals repeat' apply And.intro
    all_goals first
    | intro j on; simpa [hN] using (access.residual hrflag j.val (by simpa [hN] using on))
    | intro j on; simpa [hN] using (access.residualOut hoflag j.val (by simpa [hN] using on))

#axiomsClean firstStream_safe
#axiomsClean secondInput_safe
#axiomsClean residual_safe
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.IOSafety
