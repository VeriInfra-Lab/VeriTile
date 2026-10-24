import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.FP
import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedModel

/-! Successful execution of the complete selected typed source for arbitrary
opaque floating arithmetic. No decode-to-Real equation is a premise. This is
the source graph after width erasure, kept distinct from native contraction. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open TypedExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

def Ready {α : Type} (f : Flags) (r : Regions) (stride : Strides)
    (block row : Nat) (s : State α) : Prop :=
  s.regs .nat [] "row" = some (fun _ => row) ∧
  s.regs .nat [block] "cols" = some (fun j => j.1.val) ∧
  s.regs .ptr [] "Y" = some (fun _ => (r.y,row*stride.y)) ∧
  (f.x1=Bool.true → s.regs .ptr [] "X1" = some (fun _ => (r.x1,row*stride.x1))) ∧
  (f.residual=Bool.true → s.regs .ptr [] "RESIDUAL" = some (fun _ => (r.residual,row*stride.residual))) ∧
  (f.residualOut=Bool.true → s.regs .ptr [] "RESIDUAL_OUT" = some (fun _ => (r.residualOut,row*stride.residualOut))) ∧
  (f.w1=Bool.true → s.regs .ptr [] "Y1" = some (fun _ => (r.y1,row*stride.y1))) ∧
  ∃ x, s.regs .real [block] "x" = some x

def stagePrefix (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) : List ComputeStmt :=
  (repaired f r stride M N eps p block hb).surfaceBody.take 10

theorem prefix_success {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (s : State α) :
    ∃ t, run (engine A) (stagePrefix f r stride M N eps p block hb) s = some t ∧
      Ready f r stride block (s.pids 0) t := by
  rcases f with ⟨rms,res,resout,bias,drop,mask,scale,x1,w1,b1⟩
  cases res <;> cases resout <;> cases x1 <;> cases w1
  all_goals simp [stagePrefix,repaired,original,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralMod,bop,natLt,
    numericLt,Cell.read,ofFloat,toFloat,Region.cast,Random.i32ArangeSize,
    State.setReg,Ready]
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
end VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
