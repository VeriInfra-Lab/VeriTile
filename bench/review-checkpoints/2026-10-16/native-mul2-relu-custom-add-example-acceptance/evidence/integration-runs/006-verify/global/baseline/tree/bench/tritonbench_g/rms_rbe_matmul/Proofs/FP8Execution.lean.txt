import bench.tritonbench_g.rms_rbe_matmul.Proofs.FP8Iteration
import bench.tritonbench_g.rms_rbe_matmul.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.ByteExecution
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- The actual prologue seeds the decoded numerical recurrence while
retaining the original raw-byte memory and the source batch/program ids. -/
theorem initialize_run (X W RMS OUT : RegionName) (s : BlockState)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat) (EPS : ℝ)
    (hu : ∀ rg o, s.undef rg o = 0) :
    ∃ t, stepStmts (rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms) s = some t ∧
      Invariant X W RMS s M N BM BN BK sxb sxm sxk swk swn srms (cdiv K BK) 0 t := by
  obtain ⟨t, he, hi⟩ := preLoop X W RMS OUT s M N sxb sxm sxk swk swn srms
    sob som son BM BN BK (cdiv K BK) K EPS hu
  rcases hi with ⟨hpids, hcle, hpb, hpm, hpn, hacc, hxsum, hxp, hw, hr, hundef, hmem⟩
  refine ⟨t, he, ?_, hmem⟩
  refine ⟨hpids, hcle, hpb, hpm, hpn, ?_, ?_, hxp, hw, hr, hundef, ?_⟩
  · simpa [accPartial, ByteLoads.view] using hacc
  · simpa [xSumPartial, ByteLoads.view] using hxsum
  · exact ByteLoads.view_memory W RMS t s hmem

/-- Full actual byte-weight execution in the real projection. Its independent
formula uses finite decoded input values; the program loads original integer
bytes, executes every source conversion and retains all unwritten raw cells. -/
theorem run (input : TypedSource.Storage) (variant : Bool)
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (hu : ∀ rg o, s.undef rg o = 0) (hXW : X ≠ W) (hXR : X ≠ RMS)
    (hbytes : Bytes s W RMS N BN BK swk swn srms (cdiv K BK))
    (hinj : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN sob som son i = outOffset s N BM BN sob som son j → i = j) :
    ∃ t, exec (TypedSource.kernel input Bool.true variant X W RMS OUT M N K
        sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS).eraseDType s = some t ∧
      (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        t.readMem OUT (outOffset s N BM BN sob som son idx) =
          rmsSpec (ByteLoads.view W RMS s) X W RMS M N BM BN sxb sxm sxk swk swn srms
            BK (cdiv K BK) K EPS idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        o ≠ outOffset s N BM BN sob som son idx) → t.mem r o = s.mem r o) := by
  obtain ⟨s0, hpre, hP0⟩ := initialize_run X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK EPS hu
  obtain ⟨final, sLoop, hLoop, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := cdiv K BK) (step := 1)
      (by omega) hP0
      (fun i st hi hinv => by
        have := step X W RMS s M N BM BN BK sxb sxm sxk swk swn srms (cdiv K BK)
          hXW hXR hbytes i st hi hinv
        simpa using this)
  have hf : final = cdiv K BK := le_antisymm hPLoop.1.2.1 hfinal
  rw [hf] at hPLoop
  have hi : rrmInvariant X W RMS (ByteLoads.view W RMS s)
      M N BM BN BK sxb sxm sxk swk swn srms (cdiv K BK) (cdiv K BK)
      {sLoop with mem := (ByteLoads.view W RMS s).mem} := by
    rcases hPLoop.1 with ⟨hpids, hcle, hpb, hpm, hpn, hacc, hxsum, hxp, hw, hr, hundef, _⟩
    exact ⟨hpids, hcle, hpb, hpm, hpn, hacc, hxsum, hxp, hw, hr, hundef, rfl⟩
  obtain ⟨t, ht, hv, hframe⟩ := RealResult.postLoop_of_registers X W RMS OUT
    (ByteLoads.view W RMS s) M N BM BN BK sxb sxm sxk swk swn srms
    sob som son (cdiv K BK) K EPS hinj sLoop hi
  refine ⟨t, ?_, hv, ?_⟩
  · have hloop : stepStmt (.forRangeDyn "_i" (.constNat 0) (RealSource.ceilDivOp K BK)
        (.constNat 1) (RealSource.loop Bool.true BM BN BK sxk swk srms)) s0 = some sLoop := by
      simp only [stepForRangeAux.forRangeDyn_unfold, evalOp_constNat,
        RealTransport.ceil_eval, Option.bind_some, Tile.scalar_data]
      simpa only [stepStmt] using hLoop
    rw [exec, ComputeKernel.toAlgKernel, RealSource.projection]
    change stepStmts (RealSource.body Bool.true variant X W RMS OUT M N K
      sxb sxm sxk swk swn srms sob som son BM BN BK EPS) s = some t
    rw [RealSource.body, RealTransport.prologue, stepStmts.append_some hpre, stepStmts.cons_some hloop,
      RealTransport.tail, ht]
  · intro r o h
    rw [hframe r o h, hPLoop.2]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.ByteExecution
