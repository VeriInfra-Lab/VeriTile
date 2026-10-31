import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedAffineValues

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem firstStream_mask_readback {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (hd : f.dropout=Bool.true) (hm : f.dropoutMask=Bool.true)
    (ht : run (engine A) (firstStream f r stride M N eps p block hb) s=some t)
    (j : TileIndex [block]) (hj : j.1.val<N) :
    t.mem r.dropoutMask (row*N+j.1.val)=Cell.mk .bool
      (keepValue A cmp ((s.mem r.seeds (row)).read .int) p j.1.val) := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
  have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
    (seed_valid ((s.mem r.seeds row).read .int) hb)
  have ho : ∀ j : Fin block, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
    fun j => hv.2 (j,PUnit.unit)
  have hu := hv.1
  simp at hu
  cases hsc : f.rowscale
  all_goals simp [firstStream,repaired,original,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralMod,FP.Structural.bop,natLt,
    numericLt,cmp.available,seed_valid,Region.cast,Random.i32ArangeSize,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hu,ho,hsc,hd,hm] at ht
  all_goals subst t
  all_goals try simp only [State.setReg_mem]
  all_goals rw [vector_readback _ _ _ _ _ _ j hj]
  all_goals simp [keepValue,randomValue,ofFloat,toFloat]
  all_goals rfl


theorem secondInput_mask_readback {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (hd : f.dropout=Bool.true) (hm : f.dropoutMask=Bool.true)
    (hxf : f.x1=Bool.true)
    (ht : run (engine A) (secondInput f r stride M N eps p block hb) s=some t)
    (j : TileIndex [block]) (hj : j.1.val<N) :
    t.mem r.dropoutMask ((M+row)*N+j.1.val)=Cell.mk .bool
      (keepValue A cmp ((s.mem r.seeds (M+row)).read .int) p j.1.val) := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
  have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
    (seed_valid ((s.mem r.seeds (M+row)).read .int) hb)
  have ho : ∀ j : Fin block, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
    fun j => hv.2 (j,PUnit.unit)
  have hu := hv.1
  simp at hu
  cases hsc : f.rowscale
  all_goals simp [secondInput,repaired,original,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralMod,FP.Structural.bop,natLt,
    numericLt,cmp.available,seed_valid,Region.cast,Random.i32ArangeSize,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hu,ho,hxf,hsc,hd,hm] at ht
  all_goals subst t
  all_goals try simp only [State.setReg_mem]
  all_goals rw [vector_readback _ _ _ _ _ _ j hj]
  all_goals simp [keepValue,randomValue,ofFloat,toFloat]
  all_goals rfl


theorem residual_readback {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (v : Values α .real [block]) (hx : s.regs .real [block] "x"=some v)
    (hoflag : f.residualOut=Bool.true)
    (ht : run (engine A) (residual f r stride M N eps p block hb) s=some t)
    (j : TileIndex [block]) (hj : j.1.val<N) :
    t.mem r.residualOut (row*stride.residualOut+j.1.val)=Cell.mk .real
      (cast A (residualCombined A f r stride s N row v j)) := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,_⟩
  cases hrflag : f.residual
  all_goals simp [residual,repaired,original,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
    numericLt,Region.cast,Random.i32ArangeSize,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hrflag,hoflag] at ht
  all_goals subst t
  all_goals rw [vector_readback _ _ _ _ _ _ j hj]
  all_goals simp [residualCombined,raw,cast,op,hrflag,hj,ofFloat,toFloat]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
