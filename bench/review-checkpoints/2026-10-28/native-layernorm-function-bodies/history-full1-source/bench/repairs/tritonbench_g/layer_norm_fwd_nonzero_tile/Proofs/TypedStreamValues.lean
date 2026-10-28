import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.TypedFormula

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem firstStream_value {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (v : Values α .real [block]) (hx : s.regs .real [block] "x"=some v)
    (ht : run (engine A) (firstStream f r stride M N eps p block hb) s=some t) :
    t.regs .real [block] "x" = some (streamValues A cmp f s r row p v) := by
  by_cases hN : N = 1
  all_goals
    rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,_⟩
    have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
      (seed_valid ((s.mem r.seeds row).read .int) hb)
    have ho : ∀ j : Fin block, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
      fun j => hv.2 (j,PUnit.unit)
    have hu := hv.1
    simp at hu
    cases hsc : f.rowscale <;> cases hd : f.dropout <;> cases hm : f.dropoutMask
    all_goals simp [firstStream,actual,Source.kernel,Source.rawKernel,Source.rawKernelOne,hN,sourceFlags,sourceRegions,sourceStrides,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,integralMod,FP.Structural.bop,natLt,
      ←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
      numericLt,cmp.available,seed_valid,Region.cast,Random.i32ArangeSize,
      hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hu,ho,hsc,hd,hm] at ht
    all_goals subst t
    all_goals try dsimp only [_root_.VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource.streamValues,
      lit,op,scalarLoad,keepValue,randomValue,cast]
    all_goals simp [hsc,hd,hm,ofFloat,toFloat,hx]
    all_goals (funext i; rcases i with ⟨i,⟨⟩⟩; norm_num [FP.Structural.bop])
    all_goals norm_num [_root_.VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource.streamValues,
      lit,op,scalarLoad,keepValue,randomValue,cast,hsc,hd,hm]
    all_goals rfl

theorem secondInput_value {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (v : Values α .real [block]) (hx : s.regs .real [block] "x"=some v)
    (ht : run (engine A) (secondInput f r stride M N eps p block hb) s=some t) :
    t.regs .real [block] "x" = some (fun j => if f.x1 then op A .add (v j)
      (streamValues A cmp f s r (M+row) p
        (fun k => raw A s r.x1 (row*stride.x1) N k.1.val) j) else v j) := by
  by_cases hN : N = 1
  all_goals
    rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,_⟩
    have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
      (seed_valid ((s.mem r.seeds (M+row)).read .int) hb)
    have ho : ∀ j : Fin block, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
      fun j => hv.2 (j,PUnit.unit)
    have hu := hv.1
    simp at hu
    cases hxf : f.x1 <;> cases hsc : f.rowscale <;> cases hd : f.dropout <;> cases hm : f.dropoutMask
    all_goals simp [secondInput,actual,Source.kernel,Source.rawKernel,Source.rawKernelOne,hN,sourceFlags,sourceRegions,sourceStrides,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,integralMod,FP.Structural.bop,natLt,
      ←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
      numericLt,cmp.available,seed_valid,Region.cast,Random.i32ArangeSize,
      hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hu,ho,hxf,hsc,hd,hm] at ht
    all_goals subst t
    all_goals simp [_root_.VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource.streamValues,
      lit,op,scalarLoad,keepValue,randomValue,cast,raw,hN,
      hxf,hsc,hd,hm,ofFloat,toFloat,hx]
    all_goals (funext i; rcases i with ⟨i,⟨⟩⟩; norm_num [FP.Structural.bop])
    all_goals rfl

theorem residual_value {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (v : Values α .real [block]) (hx : s.regs .real [block] "x"=some v)
    (ht : run (engine A) (residual f r stride M N eps p block hb) s=some t) :
    t.regs .real [block] "x" = some (fun j => if f.residual then op A .add (v j)
      (raw A s r.residual (row*stride.residual) N j.1.val) else v j) := by
  by_cases hN : N = 1
  all_goals
    rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,_⟩
    cases hrflag : f.residual <;> cases hoflag : f.residualOut
    all_goals simp [residual,actual,Source.kernel,Source.rawKernel,Source.rawKernelOne,hN,sourceFlags,sourceRegions,sourceStrides,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      ←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
      numericLt,Region.cast,Random.i32ArangeSize,
      hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hrflag,hoflag] at ht
    all_goals subst t
    all_goals simp [op,raw,hN,hrflag,hoflag,ofFloat,toFloat,hx]
    all_goals (funext i; rcases i with ⟨i,⟨⟩⟩; norm_num [FP.Structural.bop])

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
