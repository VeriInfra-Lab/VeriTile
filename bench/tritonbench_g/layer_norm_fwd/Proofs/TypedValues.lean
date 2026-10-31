import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedSource

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def raw {α : Type} [Inhabited α] (A : Algebra α) (s : State α)
    (region : RegionName) (base N col : Nat) : α :=
  (engine A).cast (some .fp32) .real .real (if col<N then (s.mem region (base+col)).read .real
    else (engine A).literal (some .fp32) .real 0)

theorem prefix_value {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (s : State α) :
    ∃ t, run (engine A) (stagePrefix f r stride M N eps p block hb) s=some t ∧
      t.regs .real [block] "x" = some (fun j => raw A s r.x ((s.pids 0)*stride.x) N j.1.val) := by
  rcases f with ⟨rms,res,resout,bias,drop,mask,scale,x1,w1,b1⟩
  cases res <;> cases resout <;> cases x1 <;> cases w1
  all_goals simp [stagePrefix,repaired,original,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralMod,bop,natLt,
    numericLt,ofFloat,toFloat,Region.cast,Random.i32ArangeSize,
    State.setReg,raw]
  all_goals (funext i; cases i; simp [FP.Structural.bop])
  all_goals norm_num

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
