import bench.tritonbench_g.llama_ff_triton.Proofs.SafetyIteration

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.AccessSafety
open VeriTile.Triton
set_option maxHeartbeats 600000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- The actual output tail is safe from active output bounds only. No bound
is imposed on the inactive columns/rows computed by the same tile. -/
theorem tail (variant : Bool) (bounds : RegionBounds) (initial s : BlockState)
    (OUT : RegionName) (M N BM BN BK K soutm soutn : Nat) (EPS : ℝ)
    (hpm : s.regs .nat [] "pid_m" = some (Tile.scalar (pidM (initial.pids 0) N BN)))
    (hpn : s.regs .nat [] "pid_n" = some (Tile.scalar (pidN (initial.pids 0) N BN)))
    (hb : ∀ i : TileIndex [BM, BN], active initial M N BM BN i →
      outOffset initial N BM BN soutm soutn i < bounds OUT) :
    Stmt.TraceSafeList bounds (RealSource.tail variant OUT M N BM BN BK K soutm soutn EPS) s := by
  cases variant
  all_goals
    simp only [RealSource.tail, Bool.false_eq_true, ↓reduceIte, ffStoreTail,
      List.map_cons, List.map_nil, List.take_succ_cons, List.take_zero,
      List.drop_succ_cons, List.drop_zero, List.cons_append, List.nil_append,
      Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
      FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      Region.cast_id, eq_mpr_eq_cast, eq_mp_eq_cast, cast_eq]
    iterate 5 (apply TraceProofs.assign_cons; (solve | simp [Op.SafeAt.eq_def]); intro)
    apply TraceProofs.assign_eval_cons
    · simp [Op.SafeAt.eq_def]
    · intro om hom
      try rw [RealTransport.output_index] at hom
      rw [offsout_eval _ BM BM (pidM (initial.pids 0) N BN) "pid_m" (by simpa using hpm), Option.some.injEq] at hom
      subst om
      apply TraceProofs.assign_eval_cons
      · simp [Op.SafeAt.eq_def]
      · intro on hon
        rw [offsout_eval _ BN BN (pidN (initial.pids 0) N BN) "pid_n" (by simpa using hpn), Option.some.injEq] at hon
        subst on
        apply TraceProofs.assign_eval_cons
        · simp [Op.SafeAt.eq_def]
        · intro ptrs hptr
          rw [outptrs_eval _ OUT BM BN soutm soutn
            (fun i => rowGlobal initial N BM BN i) (fun j => colGlobal initial N BN j)
            (by simp [rowGlobal]) (by simp [colGlobal]), Option.some.injEq] at hptr
          subst ptrs
          apply TraceProofs.assign_eval_cons
          · simp [Op.SafeAt.eq_def]
          · intro masks hmasks
            rw [outmask_eval _ M N BM BN
              (fun i => rowGlobal initial N BM BN i) (fun j => colGlobal initial N BN j)
              (by simp [rowGlobal]) (by simp [colGlobal]), Option.some.injEq] at hmasks
            subst masks
            apply Stmt.TraceSafeList.cons_intro
            · simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, MemAccess.SafeAt.eq_def,
                MaskOpt.SafeAt.eq_def, true_and, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
                Op.PointerAddressesSafeOn, Op.MemorySafe.eq_def, MaskOpt.Active]
              intro ptrs hp i hi
              simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
                not_false_eq_true, BlockState.setReg_same] at hp
              cases hp
              rcases hi with ⟨masks, hm, hi⟩
              simp only [evalOp_ref, BlockState.setReg_same] at hm
              cases hm
              exact hb i (by simpa [active] using hi)
            · intro _ _
              exact Stmt.TraceSafeList.nil_intro


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.AccessSafety
