import bench.tritonbench_g.attention_score.Sources.AlgorithmProjection
import bench.tritonbench_g.attention_score.Proofs.Reused.RealExpressions
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite

/-! Arithmetic between the actual Q/M loads and pointer advances. Source
correspondence keeps all window and query-tail branches. Execution from input
memory, loop induction and flat IO are separate obligations. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealArithmetic
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- The nine arithmetic statements in each real-erased source iteration. -/
def body (BN BD N_CTX : Nat) (offset size : Int)
    (window complement evenM candidate : Bool) : List Stmt :=
  [
    Stmt.assign TileDType.real [BN, BN] "qk" (Op.full [BN, BN] (Op.const 0)),
    Stmt.assign TileDType.real [BN, BN] "qk"
      (Op.add NumericDType.real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref TileDType.real [BN, BN] "qk")
        (Op.dot (batch := []) (Op.ref TileDType.real [BN, BD] "q") (Op.ref TileDType.real [BD, BN] "k"))),
    Stmt.assign TileDType.real [BN, BN] "qk"
      (if candidate then
        Op.mul NumericDType.real Broadcast.scalarL (Op.ref TileDType.real [] "qk_scale")
          (Op.ref TileDType.real [BN, BN] "qk")
       else Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BN, BN] "qk")
          (Op.ref TileDType.real [] "qk_scale")),
    Stmt.ifThen (Op.constBool window)
      [Stmt.assign TileDType.int [BN, BN] "dist" (RealExpressions.distanceExpr BN BN offset),
        Stmt.ifThenElse (Op.constBool complement)
          [Stmt.assign TileDType.bool [BN, BN] "mask"
              (Op.ge ComparableDType.int Broadcast.scalarR (Op.ref TileDType.int [BN, BN] "dist")
                (Op.constInt size))]
          [Stmt.assign TileDType.bool [BN, BN] "mask"
              (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (Op.ge ComparableDType.int Broadcast.scalarR (Op.ref TileDType.int [BN, BN] "dist")
                  (Op.castNatToInt (Op.constNat 0)))
                (Op.lt ComparableDType.int Broadcast.scalarR (Op.ref TileDType.int [BN, BN] "dist")
                  (Op.constInt size)))]],
    Stmt.assign TileDType.real [BN, BN] "qk"
      (Op.sub NumericDType.real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref TileDType.real [BN, BN] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.real [BN] "m"))),
    Stmt.assign TileDType.real [BN, BN] "p" (Op.exp2 (Op.ref TileDType.real [BN, BN] "qk")),
    Stmt.ifThen (Op.constBool window)
      [Stmt.assign TileDType.real [BN, BN] "p"
          (Op.where (Op.ref TileDType.bool [BN, BN] "mask") (Op.ref TileDType.real [BN, BN] "p")
            (Op.broadcast (Op.const 0) [BN, BN]))],
    Stmt.ifThen (Op.boolNot (Op.constBool evenM))
      [Stmt.assign TileDType.real [BN, BN] "p"
          (Op.where
            (Op.remap [BN, BN] (Broadcast.leftIndex (Broadcast.consSame (Broadcast.consL Broadcast.nil)))
              (Op.expandDim ⟨1, by simp⟩
                (Op.lt ComparableDType.int Broadcast.scalarR
                  (Op.add NumericDType.int Broadcast.scalarR (Op.castNatToInt (Op.arange BN))
                    (Op.ref TileDType.int [] "start_m"))
                  (Op.castNatToInt (Op.constNat N_CTX)))))
            (Op.ref TileDType.real [BN, BN] "p") (Op.broadcast (Op.const 0) [BN, BN]))],
    Stmt.assign TileDType.real [BN] "o"
      (Op.add NumericDType.real (Broadcast.consSame Broadcast.nil) (Op.ref TileDType.real [BN] "o")
        (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref TileDType.real [BN, BN] "p")))
  ]

/-- Extract the actual dynamic loop, rather than assuming an arithmetic input
or a successful intermediate result. -/
def loopBody? : List Stmt → Option (List Stmt)
  | [] => none
  | .forRangeDyn "start_m" _ _ _ body :: _ | .forRangeInt "start_m" _ _ _ body :: _ => some body
  | _ :: tail => loopBody? tail

theorem loopBody_erased (code : List Stmt) :
    loopBody? (Stmt.eraseDTypeList code) = (loopBody? code).map Stmt.eraseDTypeList := by
  induction code with
  | nil => simp [Stmt.eraseDTypeList,loopBody?]
  | cons st rest ih =>
      cases st <;> try simp [Stmt.eraseDTypeList,Stmt.eraseDType.eq_def,loopBody?,ih]
      all_goals
        rename_i idx start stop stride body
        by_cases h : idx="start_m"
        · subst idx
          simp [Stmt.eraseDTypeList,Stmt.eraseDType.eq_def,loopBody?]
        · simp [Stmt.eraseDTypeList,Stmt.eraseDType.eq_def,loopBody?,h,ih]


section
variable (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN : Bool)
    (hBlock : BM = BN)

theorem original_arithmetic :
    (loopBody? (TypedSource.original Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body).map
        (fun code => (code.drop 3).take 9) =
      some (body BN BD N offset size window complement evenM Bool.false) := by
  subst BM
  rw [SourceAlgorithm.erased_body _ (SourceAlgorithm.projection _ ⟨_, rfl⟩)]
  rw [loopBody_erased]
  simp only [TypedSource.original,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind,
    loopBody?,Option.map,List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero]
  simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType,
    MaskOpt.eraseDType,VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,body,Bool.false_eq_true,↓reduceIte]
  rfl

theorem candidate_arithmetic :
    (loopBody? (TypedSource.scalarLeft Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body).map
        (fun code => (code.drop 3).take 9) =
      some (body BN BD N offset size window complement evenM Bool.true) := by
  subst BM
  rw [SourceAlgorithm.erased_body _ (SourceAlgorithm.projection _ ⟨_, rfl⟩)]
  rw [loopBody_erased]
  simp only [TypedSource.scalarLeft,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind,
    loopBody?,Option.map,List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero]
  simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType,
    MaskOpt.eraseDType,VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,body,Bool.false_eq_true,↓reduceIte]
  rfl

end
/-- Execute the arithmetic using the actual loaded operands. These are an
intermediate-state invariant, not the public kernel's input contract. -/
theorem executes (s : BlockState) (B D N startM startN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM : Bool)
    (q : Tile .real [B,D]) (k : Tile .real [D,B]) (m old : Tile .real [B])
    (hq : s.regs .real [B,D] "q" = some q)
    (hk : s.regs .real [D,B] "k" = some k)
    (hm : s.regs .real [B] "m" = some m)
    (ho : s.regs .real [B] "o" = some old)
    (hs : s.regs .real [] "qk_scale" = some (Tile.scalar (some scale)))
    (hsm : s.regs .int [] "start_m" = some (Tile.scalar (startM:Int)))
    (hsn : s.regs .int [] "start_n" = some (Tile.scalar (startN:Int))) :
    ∃ t, stepStmts (body B D N offset size window complement evenM Bool.false) s = some t ∧
      t.mem = s.mem ∧ t.pids = s.pids ∧ t.undef = s.undef := by
  cases window <;> cases complement <;> cases evenM
  all_goals
    simp [body, Bool.false_eq_true, Bool.true_eq_false, ↓reduceIte,
      stepStmts, stepStmt, evalOp.eq_def, hq, hk, hm, ho, hs, hsm, hsn,
      bind, Option.bind, RealExpressions.distanceExpr]
    repeat constructor

private theorem real_mul_commute (a b : WithBot ℝ) :
    WithBot.realMul a b = WithBot.realMul b a := by
  cases a <;> cases b
  · rfl
  · rfl
  · rfl
  · simp [WithBot.realMul, mul_comm]

private theorem eval_scaling_commute (s : BlockState) (B : Nat) :
    evalOp (.mul .real Broadcast.scalarL (.ref .real [] "qk_scale")
      (.ref .real [B,B] "qk")) s =
    evalOp (.mul .real Broadcast.scalarR (.ref .real [B,B] "qk")
      (.ref .real [] "qk_scale")) s := by
  simp only [evalOp_mul, evalOp_ref]
  cases hq : s.regs .real [B,B] "qk" <;> cases hs : s.regs .real [] "qk_scale"
  all_goals simp only [Option.bind_eq_bind, Option.bind_none, Option.bind_some]
  rename_i q scale
  congr 1
  apply Tile.ext
  intro i
  exact real_mul_commute _ _

/-- Real commutation preserves every arithmetic state and failure, independently
of experimental FP admissions. The complete-source bridge uses these bodies. -/
theorem candidate_execution (s : BlockState) (B D N : Nat) (offset size : Int)
    (window complement evenM : Bool) :
    stepStmts (body B D N offset size window complement evenM Bool.true) s =
      stepStmts (body B D N offset size window complement evenM Bool.false) s := by
  simp only [body, Bool.false_eq_true, ↓reduceIte, stepStmts, stepStmt, eval_scaling_commute]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealArithmetic
