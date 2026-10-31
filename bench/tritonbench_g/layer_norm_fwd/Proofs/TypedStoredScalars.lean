import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedNormValues
import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedFrameTail

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem normalization_mean_readback {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (v : Values α .real [block]) (hx : s.regs .real [block] "x"=some v)
    (hrms : f.rms=Bool.false) (hsep : r.mean ≠ r.rstd)
    (ht : run (engine A) (normalization f r stride M N eps p block hb) s=some t) :
    t.mem r.mean row=Cell.mk .real (cast A (meanValue A N v)) := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,_⟩
  simp [normalization,repaired,original,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
    numericLt,Region.cast,Random.i32ArangeSize,TileShape.axisDim,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hrms] at ht
  subst t
  rw [State.write_other _ r.rstd r.mean _ _ _ (Or.inl hsep)]
  simp [meanValue,lit,op,cast,ofFloat,toFloat,FP.Structural.bop]
  rfl

theorem normalization_preserves {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (ht : run (engine A) (normalization f r stride M N eps p block hb) s=some t)
    (region : RegionName) (offset : Nat) (hstd : region ≠ r.rstd)
    (hm : f.rms=Bool.false → region ≠ r.mean) :
    t.mem region offset=s.mem region offset := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
  cases hrms : f.rms
  all_goals simp [normalization,repaired,original,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
    numericLt,Region.cast,Random.i32ArangeSize,TileShape.axisDim,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hrms] at ht
  all_goals subst t
  all_goals rw [State.write_other _ r.rstd region _ _ _ (Or.inl hstd)]
  all_goals try simp only [State.setReg_mem]
  all_goals first | rfl | exact State.write_other _ r.mean region _ _ _ (Or.inl (hm hrms))

theorem affine_preserves {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : NormalizedReady f r stride block row s)
    (ht : run (engine A) (affine f r stride M N eps p block hb) s=some t)
    (region : RegionName) (offset : Nat) (hysep : region ≠ r.y)
    (hy1sep : f.w1=Bool.true → region ≠ r.y1) :
    t.mem region offset=s.mem region offset := by
  rcases hs with ⟨⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩,⟨std,hstd⟩,hm⟩
  cases hrms : f.rms
  · obtain ⟨mean,hmean⟩ := hm hrms
    cases hbias : f.bias <;> cases hw1 : f.w1 <;> cases hb1 : f.b1
    all_goals simp [affine,repaired,original,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hmean,hrms,hbias,hw1,hb1] at ht
    all_goals subst t
    all_goals first
      | exact vector_frame _ _ _ _ _ _ _ _ (by simp [hysep])
      | rw [vector_frame r.y1 region _ _ _ _ _ _ (by simp [hy1sep hw1])]
        simp only [State.setReg_mem]
        exact vector_frame _ _ _ _ _ _ _ _ (by simp [hysep])
  · cases hbias : f.bias <;> cases hw1 : f.w1 <;> cases hb1 : f.b1
    all_goals simp [affine,repaired,original,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hrms,hbias,hw1,hb1] at ht
    all_goals subst t
    all_goals first
      | exact vector_frame _ _ _ _ _ _ _ _ (by simp [hysep])
      | rw [vector_frame r.y1 region _ _ _ _ _ _ (by simp [hy1sep hw1])]
        simp only [State.setReg_mem]
        exact vector_frame _ _ _ _ _ _ _ _ (by simp [hysep])

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
