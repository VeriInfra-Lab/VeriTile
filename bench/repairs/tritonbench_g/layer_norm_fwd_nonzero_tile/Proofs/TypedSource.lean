import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.RawSource
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WholeReduction
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.TypedSupport

/-! Successful execution of the complete selected typed source for arbitrary
opaque floating arithmetic. No decode-to-Real equation is a premise. This is
the source graph after width erasure, kept distinct from native contraction. -/
namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

def Ready {α : Type} (f : Flags) (r : Regions) (stride : Strides)
    (block row : Nat) (s : State α) : Prop :=
  s.regs .int [] "row" = some (fun _ => (row : Int)) ∧
  s.regs .nat [block] "cols" = some (fun j => j.1.val) ∧
  s.regs .ptr [] "Y" = some (fun _ => (r.y,row*stride.y)) ∧
  (f.x1=Bool.true → s.regs .ptr [] "X1" = some (fun _ => (r.x1,row*stride.x1))) ∧
  (f.residual=Bool.true → s.regs .ptr [] "RESIDUAL" = some (fun _ => (r.residual,row*stride.residual))) ∧
  (f.residualOut=Bool.true → s.regs .ptr [] "RESIDUAL_OUT" = some (fun _ => (r.residualOut,row*stride.residualOut))) ∧
  (f.w1=Bool.true → s.regs .ptr [] "Y1" = some (fun _ => (r.y1,row*stride.y1))) ∧
  ∃ x, s.regs .real [block] "x" = some x

def stagePrefix (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) : List ComputeStmt :=
  (actual f r stride M N eps p block hb).surfaceBody.take 9

theorem prefix_success {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (s : State α) :
    ∃ t, run (engine A) (stagePrefix f r stride M N eps p block hb) s = some t ∧
      Ready f r stride block (s.pids 0) t := by
  by_cases hN : N = 1
  all_goals rcases f with ⟨rms,res,resout,bias,drop,mask,scale,x1,w1,b1⟩
  all_goals cases res <;> cases resout <;> cases x1 <;> cases w1
  all_goals simp [stagePrefix,actual,Source.kernel,Source.rawKernel,Source.rawKernelOne,hN,sourceFlags,sourceRegions,sourceStrides,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralMod,bop,natLt,
    numericLt,Cell.read,ofFloat,toFloat,Region.cast,Random.i32ArangeSize,
    State.setReg,Ready,←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add]
  all_goals (repeat' apply And.intro)
  all_goals (funext i; cases i; simp [FP.Structural.bop])

theorem seed_valid (seed : Int) (hb : Random.ArangeSizeDomain block) :
    (⟨.u32,.i32,7⟩ : Random.SourceProfile).inputsValid (shape := [block])
      ((seed % (2^32 : Int)).toNat : Int) (fun j => (j.1.val : Int)) = Bool.true := by
  rw [Int.toNat_of_nonneg (Int.emod_nonneg seed (by norm_num : (2^32 : Int) ≠ 0))]
  exact Domain.random_inputs_valid seed hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
