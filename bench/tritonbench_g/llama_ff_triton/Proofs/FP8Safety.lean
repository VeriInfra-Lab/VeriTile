import bench.tritonbench_g.llama_ff_triton.Proofs.FP8Execution
import bench.tritonbench_g.llama_ff_triton.Proofs.Safety

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.ByteExecution
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Actual read safety follows the raw state's four pointer registers. -/
theorem iteration_safe (bounds : RegionBounds) (A W1 W3 RMS : RegionName) (initial s : BlockState)
    (M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms t : Nat)
    (ht : t < cdiv K BK)
    (h : Invariant A W1 W3 RMS initial M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms (cdiv K BK) t s)
    (hb : AccessSafety.Reads bounds initial A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms) :
    Stmt.TraceSafeList bounds (RealSource.loop Bool.true BM BN BK sak sw1k sw3k srms)
      (s.setReg "_i" .nat [] (Tile.scalar t)) := by
  obtain ⟨_, _, _, _, _, _, _, ha, h1, h3, hr, _, _⟩ := h.1
  apply AccessSafety.iteration Bool.true bounds _ BM BN BK sak sw1k sw3k srms _ _ _ _
    (by simpa [ByteLoads.view] using ha) (by simpa [ByteLoads.view] using h1)
    (by simpa [ByteLoads.view] using h3) (by simpa [ByteLoads.view] using hr)
  · intro i
    simpa [ByteLoads.view, rowIndex, rowGlobal, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.1 t ht i.1 i.2.1
  · intro i
    simpa [ByteLoads.view, colIndex, colGlobal, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.2.1 t ht i.1 i.2.1
  · intro i
    simpa [ByteLoads.view, colIndex, colGlobal, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.2.2.1 t ht i.1 i.2.1
  · intro i
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hb.2.2.2 t ht i.2.1

/-- Full raw-byte source trace safety on the same finite-input domain as the
independent real formula, with all ceil-K and inactive-tile input reads bounded. -/
theorem complete (variant : Bool) (bounds : RegionBounds)
    (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ)
    (hu : ∀ rg o, s.undef rg o = 0)
    (hA1 : A ≠ W1) (hA3 : A ≠ W3) (hAR : A ≠ RMS)
    (hbytes : Bytes s W1 W3 RMS N BN BK sw1k sw1n sw3k sw3n srms (cdiv K BK))
    (hb : AccessSafety.Reads bounds s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN soutm soutn i < bounds OUT) :
    (TypedSource.kernel Bool.true variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨s0, hpre, hP0⟩ := initialize_run A W1 W3 OUT RMS s M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS hu
  obtain ⟨final, sLoop, hLoop, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := cdiv K BK) (step := 1)
      (by omega) hP0
      (fun i st hi hinv => by
        have := step A W1 W3 RMS s M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms (cdiv K BK)
          hA1 hA3 hAR hbytes i st hi hinv
        simpa using this)
  have hf : final = cdiv K BK := le_antisymm hPLoop.1.2.1 hfinal
  rw [hf] at hPLoop
  have hloop : stepStmt (.forRangeDyn "_i" (.constNat 0) (RealSource.ceilDivOp K BK)
      (.constNat 1) (RealSource.loop Bool.true BM BN BK sak sw1k sw3k srms)) s0 = some sLoop := by
    simp only [stepForRangeAux.forRangeDyn_unfold, evalOp_constNat,
      RealTransport.ceil_eval, Option.bind_some, Tile.scalar_data]
    simpa only [stepStmt] using hLoop
  rw [ComputeKernel.toAlgKernel, RealSource.projection]
  change Stmt.TraceSafeList bounds (RealSource.body Bool.true variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS) s
  rw [RealSource.body, RealTransport.prologue]
  apply TraceProofs.append_safe _ s (AccessSafety.prologue bounds s A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms)
  intro t ht
  have he : t = s0 := Option.some.inj (ht.symm.trans hpre)
  subst t
  rw [TraceProofs.cons_of_run_iff hloop]
  refine ⟨?_, AccessSafety.tail variant bounds s sLoop OUT M N BM BN BK K soutm soutn EPS hPLoop.1.2.2.1 hPLoop.1.2.2.2.1 ho⟩
  rw [Stmt.TraceSafe]
  refine ⟨by simp [Op.SafeAt.eq_def], by simp [RealSource.ceilDivOp, Op.SafeAt.eq_def],
    by simp [Op.SafeAt.eq_def], ?_⟩
  simp only [evalOp_constNat, RealTransport.ceil_eval, Tile.scalar_data]
  apply TraceProofs.range_safe (P := Invariant A W1 W3 RMS s M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms (cdiv K BK)) ?_ ?_ 0 s0 hP0
  · intro i st hi hinv
    exact iteration_safe bounds A W1 W3 RMS s st M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms i hi hinv hb
  · intro i st st' hi hinv hr
    obtain ⟨u, he, hu⟩ := step A W1 W3 RMS s M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms (cdiv K BK)
      hA1 hA3 hAR hbytes i st hi hinv
    have ht : st' = u := Option.some.inj (hr.symm.trans he)
    subst st'
    exact hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.ByteExecution
