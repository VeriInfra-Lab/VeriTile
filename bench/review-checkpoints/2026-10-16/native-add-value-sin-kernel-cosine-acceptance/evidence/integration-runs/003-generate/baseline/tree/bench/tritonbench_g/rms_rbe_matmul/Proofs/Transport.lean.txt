import bench.tritonbench_g.rms_rbe_matmul.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.RealTransport
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Floating storage erasure removes only the real-to-real cast. The native
integer-power operation remains present in both exact-real programs. -/
theorem real_cast {shape : TileShape} (e : Op .real shape) (s : BlockState) :
    evalOp (dtype := .real) (.castFloat .real .real e) s = evalOp e s := by
  erw [evalOp_castFloat .real .real e s]
  cases h : evalOp e s <;> simp [h, FloatDType.cast]

theorem loop (BM BN BK sxk swk srms : Nat) (s : BlockState) :
    stepStmts (RealSource.loop Bool.false BM BN BK sxk swk srms) s =
      stepStmts (rrmLoopBody BM BN BK sxk swk srms) s := by
  simp only [RealSource.loop, rrmLoopBody, List.map_cons, List.map_nil,
    List.flatMap_cons, List.flatMap_nil, List.cons_append, List.nil_append,
    RealSource.weightLoad, Bool.false_and, Bool.false_eq_true, ↓reduceIte,
    Stmt.eraseDType, Op.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    IntegralDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType,
    stepStmts, stepStmt]
  simp only [evalOp_add, evalOp_libdevicePowi, real_cast,
    eq_mpr_eq_cast, eq_mp_eq_cast, cast_eq]

theorem output_index (BM : Nat) (s : BlockState) :
    evalOp (.add .nat .scalarL (.mul .nat .nil (.constNat BM) (.ref .nat [] "pid_m")) (.arange BM)) s =
      evalOp (.add .nat .scalarL (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat BM)) (.arange BM)) s := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, evalOp_arange]
  cases h : s.regs .nat [] "pid_m" with
  | none => rfl
  | some t =>
    change some _ = some _
    congr 1
    ext i
    simp [Tile.bop, NumericDType.mul, Nat.mul_comm]

theorem tail (variant : Bool) (OUT : RegionName)
    (M N BM BN BK K sob som son : Nat) (EPS : ℝ) (s : BlockState) :
    stepStmts (RealSource.tail variant OUT M N BM BN BK K sob som son EPS) s =
      stepStmts (RealSource.tail Bool.false OUT M N BM BN BK K sob som son EPS) s := by
  cases variant
  · rfl
  · let old := (rrmStoreTail OUT M N BM BN BK K sob som son EPS).map Stmt.eraseDType
    change stepStmts (old.take 3 ++
      [.assign .nat [BM] "offs_m" (.add .nat .scalarL
        (.mul .nat .nil (.constNat BM) (.ref .nat [] "pid_m")) (.arange BM))] ++ old.drop 4) s = _
    have he : RealSource.tail Bool.false OUT M N BM BN BK K sob som son EPS = old.take 3 ++
      [.assign .nat [BM] "offs_m" (.add .nat .scalarL
        (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat BM)) (.arange BM))] ++ old.drop 4 := by
      simp [RealSource.tail, old, rrmStoreTail, Stmt.eraseDType,
        Op.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
        VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
        ComparableDType.eraseDType, TileShape.eraseAxis]
    rw [he]
    apply ProgramRewrite.context
    intro t
    simp only [stepStmts, stepStmt, output_index]

/-- Reference execution retains the actual ceiling-divided trip count and the
logical K divisor, including every unmasked padding lane. -/
def body (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat) (EPS : ℝ) : List Stmt :=
  rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms ++
    .forRange "_i" 0 (cdiv K BK) 1 (rrmLoopBody BM BN BK sxk swk srms) ::
      RealSource.tail Bool.false OUT M N BM BN BK K sob som son EPS

theorem prologue (X W RMS : RegionName)
    (M N BM BN BK sxb sxm sxk swk swn srms : Nat) :
    (rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms).map Stmt.eraseDType =
      rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms := by
  simp [rrmPrologue, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, Region.cast_id,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, FloatDType.toTileDType]

@[simp] theorem ceil_eval (K BK : Nat) (s : BlockState) :
    evalOp (RealSource.ceilDivOp K BK) s = some (Tile.scalar (cdiv K BK)) := by
  simp [RealSource.ceilDivOp, evalOp, Tile.bop, cdiv, NumericDType.div,
    NumericDType.add, NumericDType.sub]

/-- Every state and failure of the full non-FP8 source agrees with the
reference execution. No K-divisibility premise is used. -/
theorem source_exec (variant : Bool) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat) (EPS : ℝ)
    (s : BlockState) :
    stepStmts (RealSource.body Bool.false variant X W RMS OUT M N K sxb sxm sxk swk swn srms sob som son BM BN BK EPS) s =
      stepStmts (body X W RMS OUT M N K sxb sxm sxk swk swn srms sob som son BM BN BK EPS) s := by
  simp only [RealSource.body, body, prologue, ProgramRewrite.append]
  apply congrArg (Option.bind (stepStmts (rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms) s))
  funext t
  simp only [stepStmts, stepForRangeAux.forRangeDyn_unfold, evalOp_constNat,
    ceil_eval, Option.bind_some, Tile.scalar_data]
  have hr := ProgramRewrite.range_congr "_i" 0 (cdiv K BK) 1
    (RealSource.loop Bool.false BM BN BK sxk swk srms)
    (rrmLoopBody BM BN BK sxk swk srms) (loop BM BN BK sxk swk srms) t
  simp only [stepStmt, hr]
  cases stepForRangeAux "_i" 0 (cdiv K BK) 1 (rrmLoopBody BM BN BK sxk swk srms) t with
  | none => rfl
  | some u => exact tail variant OUT M N BM BN BK K sob som son EPS u

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.RealTransport
