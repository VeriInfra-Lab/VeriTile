import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.TypedNormValues
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.TypedInputValues

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem vector_readback {α : Type} (dest : RegionName) (base N block : Nat)
    (val : TileIndex [block] → Cell α) (s : State α)
    (j : TileIndex [block]) (hj : j.1.val<N) :
    ((TileShape.allIndices [block]).foldl (fun t k => if k.1.val<N then
      t.write dest (base+k.1.val) (val k) else t) s).mem dest (base+j.1.val)=val j := by
  apply State.masked_scatter_readback _ _ _ _ _ j hj
  intro a b he
  rcases a with ⟨a,⟨⟩⟩
  rcases b with ⟨b,⟨⟩⟩
  congr 1
  apply Fin.ext
  exact Nat.add_left_cancel he

/-- The first affine output is read from its actual masked store. Later Y1
stores preserve it because the wrapper supplies distinct output allocations. -/
theorem affine_y_readback {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (v : Values α .real [block]) (hx : s.regs .real [block] "x"=some v)
    (hstd : s.regs .real [] "rstd"=some (fun _ => rstdValue A f.rms N eps v))
    (hm : f.rms=Bool.false → s.regs .real [] "mean"=some (fun _ => meanValue A N v))
    (hysep : r.y ≠ r.y1)
    (ht : run (engine A) (affine f r stride M N eps p block hb) s=some t)
    (j : TileIndex [block]) (hj : j.1.val<N) :
    t.mem r.y (row*stride.y+j.1.val)=Cell.mk .real
      (affineValue A f.bias (normalizedValues A f.rms N eps v j)
        (raw A s r.weight 0 N j.1.val) (raw A s r.bias 0 N j.1.val)) := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,_⟩
  cases hrms : f.rms
  · have hmean := hm hrms
    cases hbias : f.bias <;> cases hw1 : f.w1 <;> cases hb1 : f.b1
    all_goals simp [affine,repaired,original,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hmean,hrms,hbias,hw1,hb1] at ht
    all_goals subst t
    all_goals try rw [vector_frame r.y1 r.y _ _ _ _ _ _ (by simp [hysep])]
    all_goals try simp only [State.setReg_mem]
    all_goals rw [vector_readback _ _ _ _ _ _ j hj]
    all_goals simp [affineValue,normalizedValues,raw,cast,op,defaultOp,hrms,hbias,hj,ofFloat,toFloat]
    all_goals try rfl
  · cases hbias : f.bias <;> cases hw1 : f.w1 <;> cases hb1 : f.b1
    all_goals simp [affine,repaired,original,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hrms,hbias,hw1,hb1] at ht
    all_goals subst t
    all_goals try rw [vector_frame r.y1 r.y _ _ _ _ _ _ (by simp [hysep])]
    all_goals try simp only [State.setReg_mem]
    all_goals rw [vector_readback _ _ _ _ _ _ j hj]
    all_goals simp [affineValue,normalizedValues,raw,cast,op,defaultOp,hrms,hbias,hj,ofFloat,toFloat]
    all_goals try rfl

/-- The optional second affine store uses its own weight and bias from the
incoming input memory, even though the first output has already been stored. -/
theorem affine_y1_readback {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (v : Values α .real [block]) (hx : s.regs .real [block] "x"=some v)
    (hstd : s.regs .real [] "rstd"=some (fun _ => rstdValue A f.rms N eps v))
    (hm : f.rms=Bool.false → s.regs .real [] "mean"=some (fun _ => meanValue A N v))
    (hsep : Domain.ReadWriteSeparated f r) (hw1 : f.w1=Bool.true)
    (ht : run (engine A) (affine f r stride M N eps p block hb) s=some t)
    (j : TileIndex [block]) (hj : j.1.val<N) :
    t.mem r.y1 (row*stride.y1+j.1.val)=Cell.mk .real
      (affineValue A f.b1 (normalizedValues A f.rms N eps v j)
        (raw A s r.weight1 0 N j.1.val) (raw A s r.bias1 0 N j.1.val)) := by
  have hwsep : r.weight1 ≠ r.y :=
    hsep r.weight1 (by simp [Domain.readRegion,hw1]) r.y (by simp [Domain.writeRegion])
  have hbsep : f.b1=Bool.true → r.bias1 ≠ r.y := by
    intro hb1
    exact hsep r.bias1 (by simp [Domain.readRegion,hw1,hb1]) r.y (by simp [Domain.writeRegion])
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,_⟩
  cases hrms : f.rms
  · have hmean := hm hrms
    cases hbias : f.bias <;> cases hb1 : f.b1
    all_goals simp [affine,repaired,original,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hmean,hrms,hbias,hw1,hb1] at ht
    all_goals subst t
    all_goals rw [vector_readback _ _ _ _ _ _ j hj]
    all_goals simp [affineValue,normalizedValues,raw,cast,op,defaultOp,hrms,hb1,hj,ofFloat,toFloat]
    all_goals try rfl
    all_goals rw [vector_frame r.y r.weight1 _ _ _ _ _ _ (by simp [hwsep])]
    all_goals try rw [vector_frame r.y r.bias1 _ _ _ _ _ _ (by simp [hbsep hb1])]
    all_goals simp only [State.setReg_mem]
    all_goals rfl
  · cases hbias : f.bias <;> cases hb1 : f.b1
    all_goals simp [affine,repaired,original,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hrms,hbias,hw1,hb1] at ht
    all_goals subst t
    all_goals rw [vector_readback _ _ _ _ _ _ j hj]
    all_goals simp [affineValue,normalizedValues,raw,cast,op,defaultOp,hrms,hb1,hj,ofFloat,toFloat]
    all_goals try rfl
    all_goals rw [vector_frame r.y r.weight1 _ _ _ _ _ _ (by simp [hwsep])]
    all_goals try rw [vector_frame r.y r.bias1 _ _ _ _ _ _ (by simp [hbsep hb1])]
    all_goals simp only [State.setReg_mem]
    all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
