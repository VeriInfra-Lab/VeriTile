import bench.tritonbench_g.llama_ff_triton.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.RealTransport
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Integer native power and real-exponent power agree only in this exact-real
projection for exponent two. This is not a floating-point rewrite rule. -/
theorem power_two (e : Op .real [BM, BK]) (s : BlockState) :
    evalOp (.libdevicePowi .scalarR e (.constInt 2)) s =
      evalOp (.pow .scalarR (.castFloat .real .real e) (.const 2)) s := by
  have hc : evalOp (dtype := .real) (.castFloat .real .real e) s = evalOp e s := by
    erw [evalOp_castFloat .real .real e s]
    cases h : evalOp e s <;> simp [h, FloatDType.cast]
  rw [evalOp_libdevicePowi, evalOp_pow, hc]
  have hn : evalOp (.constInt 2) s = some (Tile.scalar 2) := by rw [evalOp.eq_def]
  simp only [hn, evalOp_const]
  cases he : evalOp e s with
  | none => rfl
  | some t =>
    change some _ = some _
    apply congrArg some
    ext i
    simp only [Tile.libdevicePowi, Tile.bop, Tile.scalar_data,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
    cases h : t.data i <;>
      simp [WithBot.realPowi, WithBot.realPow, h, Real.rpow_two, zpow_ofNat, WithBot.map]

theorem loop (BM BN BK sak sw1k sw3k srms : Nat) (s : BlockState) :
    stepStmts (RealSource.loop Bool.false BM BN BK sak sw1k sw3k srms) s =
      stepStmts (ffLoopBody BM BN BK sak sw1k sw3k srms) s := by
  simp only [RealSource.loop, ffLoopBody, List.map_cons, List.map_nil,
    List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
    List.cons_append, List.nil_append, List.flatMap_cons, List.flatMap_nil,
    RealSource.weightLoad, Bool.false_and, Bool.false_eq_true, ↓reduceIte,
    Stmt.eraseDType, Op.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    IntegralDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType,
    stepStmts, stepStmt]
  simp only [evalOp_add, power_two]

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

theorem tail (variant : Bool) (OUT : RegionName) (M N BM BN BK K soutm soutn : Nat) (EPS : ℝ) (s : BlockState) :
    stepStmts (RealSource.tail variant OUT M N BM BN BK K soutm soutn EPS) s =
      stepStmts (RealSource.tail Bool.false OUT M N BM BN BK K soutm soutn EPS) s := by
  cases variant
  · rfl
  · let old := (ffStoreTail OUT M N BM BN BK K soutm soutn EPS).map Stmt.eraseDType
    change stepStmts (old.take 5 ++
      [.assign .nat [BM] "offs_outm" (.add .nat .scalarL
        (.mul .nat .nil (.constNat BM) (.ref .nat [] "pid_m")) (.arange BM))] ++ old.drop 6) s = _
    have he : RealSource.tail Bool.false OUT M N BM BN BK K soutm soutn EPS = old.take 5 ++
      [.assign .nat [BM] "offs_outm" (.add .nat .scalarL
        (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat BM)) (.arange BM))] ++ old.drop 6 := by
      simp [RealSource.tail, old, ffStoreTail, Stmt.eraseDType,
        Op.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
        VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
        ComparableDType.eraseDType, TileShape.eraseAxis]
    rw [he]
    apply ProgramRewrite.context
    intro t
    simp only [stepStmts, stepStmt, output_index]

/-- A reference execution for the actual ceiling trip count. Its floating
power is justified by `power_two`; the actual source keeps native powi. -/
def body (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ) : List Stmt :=
  ffPrologue A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms ++
    .forRange "_i" 0 (cdiv K BK) 1 (ffLoopBody BM BN BK sak sw1k sw3k srms) ::
      RealSource.tail Bool.false OUT M N BM BN BK K soutm soutn EPS


/-- The real prologue has no precision-dependent observable operation. -/
theorem prologue (A W1 W3 RMS : RegionName)
    (M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms : Nat) :
    (ffPrologue A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms).map Stmt.eraseDType =
      ffPrologue A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms := by
  simp [ffPrologue, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, Region.cast_id,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, FloatDType.toTileDType]

@[simp] theorem ceil_eval (K BK : Nat) (s : BlockState) :
    evalOp (RealSource.ceilDivOp K BK) s = some (Tile.scalar (cdiv K BK)) := by
  simp [RealSource.ceilDivOp, evalOp, Tile.bop, cdiv, NumericDType.div,
    NumericDType.add, NumericDType.sub]

/-- Preserve every state and failure of the full fp16 source. The source loop
uses its actual ceil-div operation; no divisibility premise is needed. -/
theorem source_exec (variant : Bool) (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ)
    (s : BlockState) :
    stepStmts (RealSource.body Bool.false variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS) s =
      stepStmts (body A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS) s := by
  simp only [RealSource.body, body, prologue, ProgramRewrite.append]
  apply congrArg (Option.bind (stepStmts (ffPrologue A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms) s))
  funext t
  simp only [stepStmts, stepForRangeAux.forRangeDyn_unfold, evalOp_constNat,
    ceil_eval, Option.bind_some, Tile.scalar_data]
  have hr := ProgramRewrite.range_congr "_i" 0 (cdiv K BK) 1
    (RealSource.loop Bool.false BM BN BK sak sw1k sw3k srms)
    (ffLoopBody BM BN BK sak sw1k sw3k srms) (loop BM BN BK sak sw1k sw3k srms) t
  simp only [stepStmt, hr]
  cases stepForRangeAux "_i" 0 (cdiv K BK) 1 (ffLoopBody BM BN BK sak sw1k sw3k srms) t with
  | none => rfl
  | some u => exact tail variant OUT M N BM BN BK K soutm soutn EPS u

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.RealTransport
