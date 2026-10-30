import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedFrame

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
noncomputable section
open scoped Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem secondInput_preserves {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (ht : run (engine A) (secondInput f r stride M N eps p block hb) s=some t) :
    ∀ region offset,
      (f.dropout=Bool.true → f.dropoutMask=Bool.true → f.x1=Bool.true →
        ¬ (region=r.dropoutMask ∧ Contract.inRow ((M+row)*N) N offset)) →
      t.mem region offset=s.mem region offset := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
  have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
    (seed_valid ((s.mem r.seeds (M+row)).read .int) hb)
  have ho : ∀ j : Fin block, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
    fun j => hv.2 (j,PUnit.unit)
  have hu := hv.1
  simp at hu
  intro region offset hm
  cases hxflag : f.x1 <;> cases hsc : f.rowscale <;> cases hd : f.dropout <;> cases hmask : f.dropoutMask
  all_goals simp [secondInput,repaired,original,ComputeKernel.surfaceBody,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralMod,FP.Structural.bop,natLt,
    numericLt,cmp.available,seed_valid,Region.cast,Random.i32ArangeSize,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hu,ho,hxflag,hsc,hd,hmask] at ht
  all_goals subst t
  all_goals try simp only [State.setReg_mem]
  all_goals first | rfl | exact vector_frame _ _ _ _ _ _ _ _ (hm hd hmask hxflag)


theorem residual_preserves {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (ht : run (engine A) (residual f r stride M N eps p block hb) s=some t) :
    ∀ region offset, region ≠ r.residualOut → t.mem region offset=s.mem region offset := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
  intro region offset hsep
  have hm : f.residualOut=Bool.true →
      ¬ (region=r.residualOut ∧ Contract.inRow (row*stride.residualOut) N offset) := by
    intro _ h; exact hsep h.1
  cases hrflag : f.residual <;> cases hoflag : f.residualOut
  all_goals simp [residual,repaired,original,ComputeKernel.surfaceBody,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
    numericLt,Region.cast,Random.i32ArangeSize,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hrflag,hoflag] at ht
  all_goals subst t
  all_goals try simp only [State.setReg_mem]
  all_goals first | rfl | exact vector_frame _ _ _ _ _ _ _ _ (hm hoflag)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
