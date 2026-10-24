import bench.tritonbench_g.llama_ff_triton.Proofs.FP8Iteration
import bench.tritonbench_g.llama_ff_triton.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.ByteExecution
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- The actual prologue seeds the raw-byte recurrence without altering any
input representation or original memory cell. -/
theorem initialize_run (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ)
    (hu : ∀ rg o, s.undef rg o = 0) :
    ∃ t, stepStmts (ffPrologue A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms) s = some t ∧
      Invariant A W1 W3 RMS s M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms (cdiv K BK) 0 t := by
  obtain ⟨t, he, hi⟩ := preLoop A W1 W3 OUT RMS s M N sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK (cdiv K BK) K EPS hu
  rcases hi with ⟨hpids, hcle, hpm, hpn, hacc1, hacc2, hasum, hap, h1, h3, hr, hundef, hmem⟩
  refine ⟨t, he, ?_, hmem⟩
  refine ⟨hpids, hcle, hpm, hpn, ?_, ?_, ?_, hap, h1, h3, hr, hundef, ?_⟩
  · simpa [accPartial, ByteLoads.view] using hacc1
  · simpa [accPartial, ByteLoads.view] using hacc2
  · simpa [aSumPartial, ByteLoads.view] using hasum
  · exact ByteLoads.view_memory W1 W3 RMS t s hmem

/-- Full actual FP8 execution and all non-output typed cells. The input view
is used only to state independent sums of decoded finite weights; the executed
program loads integer bytes and preserves their original encoding. -/
theorem run (variant : Bool) (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ)
    (hu : ∀ rg o, s.undef rg o = 0)
    (hA1 : A ≠ W1) (hA3 : A ≠ W3) (hAR : A ≠ RMS)
    (hbytes : Bytes s W1 W3 RMS N BN BK sw1k sw1n sw3k sw3n srms (cdiv K BK))
    (hinj : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN soutm soutn i = outOffset s N BM BN soutm soutn j → i = j) :
    ∃ t, exec (TypedSource.kernel Bool.true variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).eraseDType s = some t ∧
      (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        t.readMem OUT (outOffset s N BM BN soutm soutn idx) =
          ffSpec (ByteLoads.view W1 W3 RMS s) A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms BK (cdiv K BK) K EPS idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        o ≠ outOffset s N BM BN soutm soutn idx) → t.mem r o = s.mem r o) := by
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
  have hi : ffInvariant A W1 W3 RMS (ByteLoads.view W1 W3 RMS s)
      M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms (cdiv K BK) (cdiv K BK)
      {sLoop with mem := (ByteLoads.view W1 W3 RMS s).mem} := by
    rcases hPLoop.1 with ⟨hpids, hcle, hpm, hpn, hacc1, hacc2, hasum, hap, h1, h3, hr, hundef, _⟩
    exact ⟨hpids, hcle, hpm, hpn, hacc1, hacc2, hasum, hap, h1, h3, hr, hundef, rfl⟩
  obtain ⟨t, ht, hv, hframe⟩ := RealResult.postLoop_of_registers A W1 W3 OUT RMS
    (ByteLoads.view W1 W3 RMS s) M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms soutm soutn (cdiv K BK) K EPS hinj sLoop hi
  refine ⟨t, ?_, hv, ?_⟩
  · have hloop : stepStmt (.forRangeDyn "_i" (.constNat 0) (RealSource.ceilDivOp K BK)
        (.constNat 1) (RealSource.loop Bool.true BM BN BK sak sw1k sw3k srms)) s0 = some sLoop := by
      simp only [stepForRangeAux.forRangeDyn_unfold, evalOp_constNat,
        RealTransport.ceil_eval, Option.bind_some, Tile.scalar_data]
      simpa only [stepStmt] using hLoop
    rw [exec, ComputeKernel.toAlgKernel, RealSource.projection]
    change stepStmts (RealSource.body Bool.true variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS) s = some t
    rw [RealSource.body, RealTransport.prologue, stepStmts.append_some hpre, stepStmts.cons_some hloop,
      RealTransport.tail, ht]
  · intro r o h
    rw [hframe r o h, hPLoop.2]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.ByteExecution
