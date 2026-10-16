import bench.tritonbench_g.llama_ff_triton.Proofs.SafetyTail

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.AccessSafety
open VeriTile.Triton
set_option maxHeartbeats 600000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Whole-source fp16 access safety follows the actual dynamic ceiling loop
and the four unmasked pointer tiles, not a vacuous zero-extent allocation. -/
theorem complete_fp16 (variant : Bool) (bounds : RegionBounds)
    (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ)
    (hu : ∀ rg o, s.undef rg o = 0)
    (hb : Reads bounds s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN soutm soutn i < bounds OUT) :
    (TypedSource.kernel Bool.false variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨s0, hpre, hP0⟩ := preLoop A W1 W3 OUT RMS s M N sam sak sw1k sw1n sw3k sw3n
    soutm soutn srms BM BN BK (cdiv K BK) K EPS hu
  have hpre' : stepStmts (ffPrologue A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms) s = some s0 := hpre
  obtain ⟨final, sLoop, hLoop, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := cdiv K BK) (step := 1)
      (by omega) hP0
      (fun t st hlt hinv => by
        have := ff_step A W1 W3 RMS s M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms
          (cdiv K BK) t st hlt hinv
        simpa using this)
  have hf : final = cdiv K BK := le_antisymm hPLoop.2.1 hfinal
  rw [hf] at hPLoop
  have hloop : stepStmt (.forRangeDyn "_i" (.constNat 0) (RealSource.ceilDivOp K BK)
      (.constNat 1) (RealSource.loop Bool.false BM BN BK sak sw1k sw3k srms)) s0 = some sLoop := by
    simp only [stepForRangeAux.forRangeDyn_unfold, evalOp_constNat,
      RealTransport.ceil_eval, Option.bind_some, Tile.scalar_data]
    rw [ProgramRewrite.range_congr _ _ _ _ _ _ (RealTransport.loop BM BN BK sak sw1k sw3k srms)]
    simpa only [stepStmt] using hLoop
  rw [ComputeKernel.toAlgKernel, RealSource.projection]
  change Stmt.TraceSafeList bounds (RealSource.body Bool.false variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS) s
  rw [RealSource.body, RealTransport.prologue]
  apply TraceProofs.append_safe _ s (prologue bounds s A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms)
  intro t ht
  have he : t = s0 := Option.some.inj (ht.symm.trans hpre')
  subst t
  rw [TraceProofs.cons_of_run_iff hloop]
  refine ⟨?_, tail variant bounds s sLoop OUT M N BM BN BK K soutm soutn EPS hPLoop.2.2.1 hPLoop.2.2.2.1 ho⟩
  rw [Stmt.TraceSafe]
  refine ⟨by simp [Op.SafeAt.eq_def], by simp [RealSource.ceilDivOp, Op.SafeAt.eq_def],
    by simp [Op.SafeAt.eq_def], ?_⟩
  simp only [evalOp_constNat, RealTransport.ceil_eval, Tile.scalar_data]
  apply TraceProofs.range_safe (P := ffInvariant A W1 W3 RMS s M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms (cdiv K BK)) ?_ ?_ 0 s0 hP0
  · intro i st hi hinv
    exact iteration_of_invariant Bool.false bounds A W1 W3 RMS s st M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms i hi hinv hb
  · intro i st st' hi hinv hr
    rw [RealTransport.loop] at hr
    obtain ⟨u, he, hu⟩ := ff_step A W1 W3 RMS s M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms (cdiv K BK) i st hi hinv
    have ht : st' = u := Option.some.inj (hr.symm.trans he)
    subst st'
    exact hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.AccessSafety
