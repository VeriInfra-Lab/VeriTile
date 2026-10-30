import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedFormula
import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedTail

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem pointwise_bop {α : Type} (fn : α → α → α)
    (x y : TileIndex [block] → α) :
    FP.Structural.bop fn Broadcast.nil.consSame x y = fun j => fn (x j) (y j) := by
  funext j
  rcases j with ⟨j,⟨⟩⟩
  rfl

theorem normalization_values {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (v : Values α .real [block]) (hx : s.regs .real [block] "x"=some v)
    (ht : run (engine A) (normalization f r stride M N eps p block hb) s=some t) :
    t.regs .real [block] "x"=some v ∧
    t.regs .real [] "rstd"=some (fun _ => rstdValue A f.rms N eps v) ∧
    (f.rms=Bool.false → t.regs .real [] "mean"=some (fun _ => meanValue A N v)) := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,_⟩
  cases hrms : f.rms
  all_goals simp [normalization,repaired,original,ComputeKernel.surfaceBody,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
    numericLt,Region.cast,Random.i32ArangeSize,TileShape.axisDim,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hrms] at ht
  all_goals subst t
  all_goals simp [rstdValue,varianceValue,meanValue,centeredValues,lit,op,
    hrms,ofFloat,toFloat,hx]
  all_goals (repeat' apply And.intro)
  all_goals (funext i; cases i; norm_num [FP.Structural.bop])
  all_goals try rw [pointwise_bop]
  all_goals rfl

theorem normalization_rstd_readback {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (v : Values α .real [block]) (hx : s.regs .real [block] "x"=some v)
    (ht : run (engine A) (normalization f r stride M N eps p block hb) s=some t) :
    t.mem r.rstd row=Cell.mk .real (cast A (rstdValue A f.rms N eps v)) := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,_⟩
  cases hrms : f.rms
  all_goals simp [normalization,repaired,original,ComputeKernel.surfaceBody,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
    numericLt,Region.cast,Random.i32ArangeSize,TileShape.axisDim,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hrms] at ht
  all_goals subst t
  all_goals simp [rstdValue,varianceValue,meanValue,centeredValues,lit,op,cast,
    hrms,ofFloat,toFloat,hx,FP.Structural.bop]
  all_goals norm_num
  all_goals rw [pointwise_bop]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
