import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Kernels
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchReference
import VeriTile.Triton.Float.StructuralSafety

/-! The complete pasted source is connected to the retained proof presentation
by execution and access-safety equalities. The only semantic changes in that
presentation are a negated sentinel test, spelling of zero, and an extra
same-format output cast. The latter is discharged by the concrete graph
backend; no cast identity is required of an arbitrary algebra. -/
namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.SourceConnection
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

private def CodeEq {α : Type} [Inhabited α] (M : Algebra α)
    (a b : List ComputeStmt) : Prop :=
  (∀ s, run M a s=run M b s) ∧
  (∀ bounds s, Safety.traceSafeList M bounds a s ↔ Safety.traceSafeList M bounds b s)
private def StmtEq {α : Type} [Inhabited α] (M : Algebra α)
    (a b : ComputeStmt) : Prop :=
  (∀ s, step M a s=step M b s) ∧
  (∀ bounds s, Safety.stepSafe M bounds a s ↔ Safety.stepSafe M bounds b s)

private theorem code_same {α : Type} [Inhabited α] (M : Algebra α)
    (a : List ComputeStmt) : CodeEq M a a := ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
private theorem stmt_same {α : Type} [Inhabited α] (M : Algebra α)
    (a : ComputeStmt) : StmtEq M a a := ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
private theorem code_cons {α : Type} [Inhabited α] {M : Algebra α}
    {a b : ComputeStmt} {as bs : List ComputeStmt}
    (head : StmtEq M a b) (tail : CodeEq M as bs) : CodeEq M (a::as) (b::bs) := by
  constructor
  · intro s
    simp only [run,head.1,tail.1]
  · intro bounds s
    simp only [Safety.traceSafeList_cons,head.1,head.2,tail.2]

private theorem if_same {α : Type} [Inhabited α] {M : Algebra α}
    (c : Op .bool []) {a b : List ComputeStmt} (body : CodeEq M a b) :
    StmtEq M (.ifThen c a) (.ifThen c b) := by
  constructor
  · intro s; simp only [step,body.1]
  · intro bounds s
    rw [Safety.stepSafe_unfold,Safety.stepSafe_unfold]
    cases evalOp M none c s <;> simp only [body.2]

private theorem if_else_same {α : Type} [Inhabited α] {M : Algebra α}
    (c : Op .bool []) {a b a' b' : List ComputeStmt}
    (yes : CodeEq M a a') (no : CodeEq M b b') :
    StmtEq M (.ifThenElse c a b) (.ifThenElse c a' b') := by
  constructor
  · intro s; simp only [step,yes.1,no.1]
  · intro bounds s
    rw [Safety.stepSafe_unfold,Safety.stepSafe_unfold]
    cases evalOp M none c s <;> simp only [yes.2,no.2]

private theorem range_run {α : Type} [Inhabited α] {M : Algebra α}
    (idx : RegName) (cur stop stride : Nat) {a b : List ComputeStmt}
    (same : CodeEq M a b) (s : State α) :
    range M idx cur stop stride a s=range M idx cur stop stride b s := by
  rw [range,range]
  by_cases hz : stride=0
  · simp [hz]
  · simp only [hz,↓reduceIte]
    by_cases hc : cur<stop
    · simp only [hc,↓reduceIte,same.1]
      cases run M b (s.setReg idx .nat [] (fun _ => cur)) with
      | none => rfl
      | some t => exact range_run idx (cur+stride) stop stride same t
    · simp only [hc,↓reduceIte]
termination_by stop-cur
decreasing_by omega

private theorem range_safe {α : Type} [Inhabited α] {M : Algebra α}
    (bounds : RegionName → Nat) (idx : RegName) (cur stop stride : Nat)
    {a b : List ComputeStmt} (same : CodeEq M a b) (s : State α) :
    Safety.rangeSafe M bounds idx cur stop stride a s ↔
      Safety.rangeSafe M bounds idx cur stop stride b s := by
  rw [Safety.rangeSafe_unfold,Safety.rangeSafe_unfold]
  by_cases hz : stride=0
  · simp [hz]
  · simp only [hz,↓reduceIte]
    by_cases hc : cur<stop
    · simp only [hc,↓reduceIte,same.1,same.2]
      cases run M b (s.setReg idx .nat [] (fun _ => cur)) with
      | none => rfl
      | some t => exact and_congr_right fun _ => range_safe bounds idx (cur+stride) stop stride same t
    · simp only [hc,↓reduceIte]
termination_by stop-cur
decreasing_by omega

private theorem range_same {α : Type} [Inhabited α] {M : Algebra α}
    (idx : RegName) (start stop stride : Op .nat []) {a b : List ComputeStmt}
    (body : CodeEq M a b) :
    StmtEq M (.forRangeDyn idx start stop stride a) (.forRangeDyn idx start stop stride b) := by
  constructor
  · intro s; simp only [step,range_run idx _ _ _ body]
  · intro bounds s
    rw [Safety.stepSafe_unfold,Safety.stepSafe_unfold]
    cases evalOp M none start s <;> cases evalOp M none stop s <;>
      cases evalOp M none stride s <;> simp only [range_safe bounds idx _ _ _ body]

private theorem sentinel_if {α : Type} [Inhabited α] {M : Algebra α}
    {a b : List ComputeStmt} (body : CodeEq M a b) :
    StmtEq M (.ifThen (.boolNot (.eq .int .nil (.ref .int [] "lora_index") (.constInt (-1)))) a)
      (.ifThen (.ne .int .nil (.ref .int [] "lora_index") (.constInt (-1))) b) := by
  have guard (s : State α) :
      evalOp M none (.boolNot (.eq .int .nil (.ref .int [] "lora_index") (.constInt (-1)))) s =
      evalOp M none (.ne .int .nil (.ref .int [] "lora_index") (.constInt (-1))) s := by
    cases he : s.regs .int [] "lora_index" <;>
      simp [evalOp_unfold,numericEq,bop,Broadcast.leftIndex,Broadcast.rightIndex,he]
    rfl
  constructor
  · intro s; simp only [step,guard,body.1]
  · intro bounds s
    rw [Safety.stepSafe_unfold,Safety.stepSafe_unfold]
    simp only [guard,body.2]
    simp [Safety.safeOp_unfold]

private theorem eval_real_cast (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n,ReductionPlan n) (precision : Option ComputeDType)
    (e : Op .real shape) (s : State Word) :
    evalOp (Branches.engine rest cfg plans) precision (.castFloat .real .real e) s =
      evalOp (Branches.engine rest cfg plans) precision e s := by
  have hc (x : Word) : (Branches.engine rest cfg plans).cast precision .real .real x=x := by
    simp [Branches.engine,FPValue.backend,FPValue.formatOf,FPValue.castWord]
  change ((evalOp (Branches.engine rest cfg plans) precision e s).bind
    fun v => some (fun i => (Branches.engine rest cfg plans).cast precision .real .real (v i))) = _
  cases he : evalOp (Branches.engine rest cfg plans) precision e s <;>
    simp only [Option.bind_none,Option.bind_some,hc]

private theorem store_cast (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n,ReductionPlan n) (rows : Nat) (mem : MemAccess .real [rows])
    (e : Op .real [rows]) (mask : MaskOpt .real [rows]) :
    StmtEq (Branches.engine rest cfg plans)
      (.store .real [rows] mem (.compute (.alg .fp32 e)) mask)
      (.store .real [rows] mem (.compute (.alg .fp32 (.castFloat .real .real e))) mask) := by
  constructor
  · intro s
    simp only [step,evalExpr,evalComputeOp]
    erw [eval_real_cast]
    rfl
  · intro bounds s
    have hc (precision : Option ComputeDType) :
        Safety.safeOp (Branches.engine rest cfg plans) precision bounds s
          (.castFloat .real .real e) ↔
        Safety.safeOp (Branches.engine rest cfg plans) precision bounds s e := by
      rw [Safety.safeOp_unfold]
      rfl
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    erw [hc]
    rfl

private theorem source_code (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n,ReductionPlan n) (p : Branches.Parameters) :
    CodeEq (Branches.engine rest cfg plans) (fullSource p).surfaceBody
      (Branches.source .singleHalf .single p).surfaceBody := by
  dsimp only [fullSource,Branches.source,
    VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source,ComputeKernel.surfaceBody]
  simp only [show (0.0 : ℝ)=0 by norm_num]
  repeat' first
    | exact code_same _ _
    | exact stmt_same _ _
    | apply code_cons
    | apply if_same
    | apply sentinel_if
    | apply if_else_same
    | apply range_same
    | exact store_cast _ _ _ _ _ _ _

/-- Equal complete executions, including failure and the entire final state. -/
theorem run_eq (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n,ReductionPlan n) (p : Branches.Parameters) (s : State Word) :
    exec (Branches.engine rest cfg plans) (fullSource p) s =
      exec (Branches.engine rest cfg plans) (Branches.source .singleHalf .single p) s :=
  (source_code rest cfg plans p).1 s

/-- Same actual access safety; execution equality alone is insufficient. -/
theorem safe_iff (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n,ReductionPlan n) (p : Branches.Parameters)
    (bounds : RegionName → Nat) (s : State Word) :
    TraceSafe (Branches.engine rest cfg plans) bounds (fullSource p) s ↔
      TraceSafe (Branches.engine rest cfg plans) bounds (Branches.source .singleHalf .single p) s :=
  (source_code rest cfg plans p).2 bounds s

/-- The logical body is shared by the two accepted signed index-storage ABIs.
Their native width/range obligations remain with the existing caller. -/
theorem index_profiles (p : Branches.Parameters) : fullSourceInt32 p=fullSource p := rfl

#axiomsClean run_eq
#axiomsClean safe_iff
#axiomsClean index_profiles

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.SourceConnection
