import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeGraph

/-! Exact source-site checks. These run the assignments extracted from the
actual translated helper body, for either helper and either retained stage.
The state variables are the registers of that invocation at the relevant
source instructions. They are not numerical identities or output premises. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.NativeGraph
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open TypedExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def assignments (name : RegName) (stmts : List ComputeStmt) : List ComputeStmt :=
  stmts.filter fun s => match s with | .assign _ _ n _ => n==name | _ => Bool.false

theorem source_score_scales (p : Config) (BD : Nat) (second : Bool) (s : State Expr)
    (q : Values Expr .real [128,BD]) (k : Values Expr .real [BD,64])
    (qs ks : Values Expr .real [])
    (hq : s.regs .real [128,BD] "q"=some q)
    (hk : s.regs .real [BD,64] "k"=some k)
    (hqs : s.regs .real [] "q_scale"=some qs)
    (hks : s.regs .real [] "k_scale"=some ks) :
    run (engine backend) ((assignments "qk" (body p BD second)).take 1) s=
      some (s.setReg "qk" .real [128,64] (fun i =>
        .product (some .fp32) .fp32 Bool.true
          (.product (some .fp32) .fp32 Bool.false
            (.scoreDot (some .fp32) "tf32" [] 128 BD 64 q k i) (qs PUnit.unit))
          (ks PUnit.unit))) := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [body,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Option.map_some,Option.getD_some,beq_self_eq_true]
  all_goals simp [assignments,hd,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,engine,Algebra.withDefaultPrecision,
    resolvePrecision,backend,hq,hk,hqs,hks,ofFloat,toFloat,cast,nativeDType,consume]
  all_goals apply congrArg (s.setReg "qk" .real [128,64])
  all_goals funext i
  all_goals rfl

def isAccumulator (stmt : ComputeStmt) : Bool :=
  match stmt with | .assign _ _ name _ => name=="acc" | _ => Bool.false

def betweenUpdates (stmts : List ComputeStmt) : List ComputeStmt :=
  (((stmts.dropWhile fun stmt => !isAccumulator stmt).drop 1).takeWhile fun stmt => !isAccumulator stmt)

theorem source_between_updates_pure (p : Config) (BD : Nat) (second : Bool) :
    TypedPreservation.Pure ["acc","alpha"] (betweenUpdates (body p BD second)) := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [body,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Option.map_some,Option.getD_some,beq_self_eq_true]
  all_goals simp [betweenUpdates,isAccumulator,hd]
  all_goals exact .assign _ _ _ _ (by simp) _ (.assign _ _ _ _ (by simp) _ .nil)

theorem source_between_updates_preserves (p : Config) (BD : Nat) (second : Bool)
    (s t : State Expr) (ht : run (engine backend) (betweenUpdates (body p BD second)) s=some t) :
    TypedPreservation.Keep ["acc","alpha"] s t :=
  TypedPreservation.pure_keep (engine backend) _ _ (source_between_updates_pure p BD second) s t ht

theorem source_denominator (p : Config) (BD : Nat) (second : Bool) (s : State Expr)
    (old alpha sum : Values Expr .real [128])
    (ho : s.regs .real [128] "l_i"=some old)
    (ha : s.regs .real [128] "alpha"=some alpha)
    (hs : s.regs .real [128] "l_ij"=some sum) :
    run (engine backend) (assignments "l_i" (body p BD second)) s=
      some (s.setReg "l_i" .real [128]
        (fun i => .fma (some .fp32) .fp32 (old i) (alpha i) (sum i))) := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [body,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Option.map_some,Option.getD_some,beq_self_eq_true]
  all_goals simp [assignments,hd,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,engine,Algebra.withDefaultPrecision,
    resolvePrecision,backend,ho,ha,hs,ofFloat,toFloat,online_update]
  all_goals apply congrArg (s.setReg "l_i" .real [128])
  all_goals funext i
  all_goals exact online_update _ _ _ _

def valueDot (BD : Nat) (prob : Values Expr .fp16 [128,64])
    (value : Values Expr .real [64,BD]) (i : TileIndex [128,BD]) : Expr :=
  .typedDot (some .fp32) .fp16 .fp16 .fp16 {outDType:=some .fp16} [] 128 64 BD
    prob (fun j => .cast (some .fp32) .fp32 .fp16 (value j)) i

/-- Both extracted assignments share the same old accumulator and alpha.
The intervening source V load and probability cast do not assign either;
`prob` and `value` are their actual post-load/post-cast register contents. -/
theorem source_accumulator (p : Config) (BD : Nat) (second : Bool) (s : State Expr)
    (old : Values Expr .real [128,BD]) (alpha : Values Expr .real [128])
    (prob : Values Expr .fp16 [128,64]) (value : Values Expr .real [64,BD])
    (ho : s.regs .real [128,BD] "acc"=some old)
    (ha : s.regs .real [128] "alpha"=some alpha)
    (hp : s.regs .fp16 [128,64] "p"=some prob)
    (hv : s.regs .real [64,BD] "v"=some value) :
    ∃ t, run (engine backend) (assignments "acc" (body p BD second)) s=some t ∧
      t.regs .real [128,BD] "acc"=some (fun i =>
        Expr.fma (some .fp32) .fp32 (old i) (alpha (i.1,PUnit.unit))
          (.cast (some .fp32) .fp16 .fp32 (valueDot BD prob value i))) := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [body,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Option.map_some,Option.getD_some,beq_self_eq_true]
  all_goals simp [assignments,hd,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,engine,Algebra.withDefaultPrecision,
    resolvePrecision,backend,ho,ha,hp,hv,ofFloat,toFloat,State.setReg,
    online_update,cast,nativeDType,valueDot]
  all_goals funext i
  all_goals exact online_update _ _ _ _

theorem source_normalization (p : Config) (BD : Nat) (s : State Expr)
    (acc : Values Expr .real [128,BD]) (denom : Values Expr .real [128])
    (ha : s.regs .real [128,BD] "acc"=some acc)
    (hd : s.regs .real [128] "l_i"=some denom) :
    run (engine backend) ((tail p BD Bool.false).take 1) s=
      some (s.setReg "acc" .real [128,BD] (fun i =>
        .binary (some .fp32) .fp32 .div (acc i) (denom (i.1,PUnit.unit)))) := by
  simp only [tail,source,Bool.false_eq_true,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero]
  simp [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,engine,Algebra.withDefaultPrecision,
    resolvePrecision,backend,ha,hd,ofFloat,toFloat,consume,nativeDType]
  apply congrArg (s.setReg "acc" .real [128,BD])
  funext i
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.NativeGraph
