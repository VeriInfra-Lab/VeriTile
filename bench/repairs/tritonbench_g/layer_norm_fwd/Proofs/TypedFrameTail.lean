import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedFrame

namespace VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open TypedExecution
noncomputable section
open scoped Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem secondInput_frame {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (ht : run (engine A) (secondInput f r stride M N eps p block hb) s=some t) :
    Frame f r stride M N row s t := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
  have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
    (seed_valid ((s.mem r.seeds (M+row)).read .int) hb)
  have ho : ∀ j : Fin block, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
    fun j => hv.2 (j,PUnit.unit)
  have hu := hv.1
  simp at hu
  intro region offset hw
  have hm : f.dropout=Bool.true → f.dropoutMask=Bool.true → f.x1=Bool.true →
      ¬ (region=r.dropoutMask ∧ Contract.inRow ((M+row)*N) N offset) := by
    intro hd hm hx h
    exact hw (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨hd,hm,hx,h⟩))))))
  cases hxflag : f.x1 <;> cases hsc : f.rowscale <;> cases hd : f.dropout <;> cases hmask : f.dropoutMask
  all_goals simp [secondInput,repaired,original,ComputeKernel.surfaceBody,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralMod,FP.Structural.bop,natLt,
    numericLt,cmp.available,seed_valid,Region.cast,Random.i32ArangeSize,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hu,ho,hxflag,hsc,hd,hmask] at ht
  all_goals subst t
  all_goals try simp only [State.setReg_mem]
  all_goals first | rfl | exact vector_frame _ _ _ _ _ _ _ _ (hm hd hmask hxflag)

theorem residual_frame {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (ht : run (engine A) (residual f r stride M N eps p block hb) s=some t) :
    Frame f r stride M N row s t := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
  intro region offset hw
  have hm : f.residualOut=Bool.true →
      ¬ (region=r.residualOut ∧ Contract.inRow (row*stride.residualOut) N offset) := by
    intro hf h
    exact hw (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨hf,h⟩)))))
  cases hrflag : f.residual <;> cases hoflag : f.residualOut
  all_goals simp [residual,repaired,original,ComputeKernel.surfaceBody,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
    numericLt,Region.cast,Random.i32ArangeSize,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hrflag,hoflag] at ht
  all_goals subst t
  all_goals try simp only [State.setReg_mem]
  all_goals first | rfl | exact vector_frame _ _ _ _ _ _ _ _ (hm hoflag)

theorem normalization_frame {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (ht : run (engine A) (normalization f r stride M N eps p block hb) s=some t) :
    Frame f r stride M N row s t := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
  intro region offset hw
  have hm : f.rms=Bool.false → ¬ (region=r.mean ∧ offset=row) := by
    intro hf h; exact hw (Or.inr (Or.inr (Or.inr (Or.inl ⟨hf,h⟩))))
  have hr : ¬ (region=r.rstd ∧ offset=row) := by
    intro h; exact hw (Or.inr (Or.inr (Or.inl h)))
  cases hrms : f.rms
  all_goals simp [normalization,repaired,original,ComputeKernel.surfaceBody,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
    numericLt,Region.cast,Random.i32ArangeSize,TileShape.axisDim,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hrms] at ht
  all_goals subst t
  all_goals try simp only [State.setReg_mem]
  all_goals rw [State.write_other _ r.rstd region row offset _ (by tauto)]
  all_goals try simp only [State.setReg_mem]
  all_goals first
    | rfl
    | exact State.write_other _ r.mean region row offset _ (by have hh := hm hrms; tauto)

theorem affine_frame {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : NormalizedReady f r stride block row s)
    (ht : run (engine A) (affine f r stride M N eps p block hb) s=some t) :
    Frame f r stride M N row s t := by
  rcases hs with ⟨⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩,⟨std,hstd⟩,hm⟩
  intro region offset hw
  have hfirst : ¬ (region=r.y ∧ Contract.inRow (row*stride.y) N offset) := by
    intro h; exact hw (Or.inr (Or.inl h))
  have hsecond : f.w1=Bool.true →
      ¬ (region=r.y1 ∧ Contract.inRow (row*stride.y1) N offset) := by
    intro hf h; exact hw (Or.inl ⟨hf,h⟩)
  cases hrms : f.rms
  · obtain ⟨mean,hmean⟩ := hm hrms
    cases hbias : f.bias <;> cases hw1 : f.w1 <;> cases hb1 : f.b1
    all_goals simp [affine,repaired,original,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hmean,hrms,hbias,hw1,hb1] at ht
    all_goals subst t
    all_goals first
      | exact vector_frame _ _ _ _ _ _ _ _ hfirst
      | rw [vector_frame _ _ _ _ _ _ _ _ (hsecond hw1)]
        simp only [State.setReg_mem]
        exact vector_frame _ _ _ _ _ _ _ _ hfirst
  · cases hbias : f.bias <;> cases hw1 : f.w1 <;> cases hb1 : f.b1
    all_goals simp [affine,repaired,original,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hrms,hbias,hw1,hb1] at ht
    all_goals subst t
    all_goals first
      | exact vector_frame _ _ _ _ _ _ _ _ hfirst
      | rw [vector_frame _ _ _ _ _ _ _ _ (hsecond hw1)]
        simp only [State.setReg_mem]
        exact vector_frame _ _ _ _ _ _ _ _ hfirst

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
