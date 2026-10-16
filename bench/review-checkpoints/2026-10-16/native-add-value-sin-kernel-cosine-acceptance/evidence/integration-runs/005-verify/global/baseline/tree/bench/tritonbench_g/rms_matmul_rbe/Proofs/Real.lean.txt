import bench.tritonbench_g.rms_matmul_rbe.Proofs.Transport
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Result

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.RealExecution
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- Complete non-FP8 source execution succeeds and returns the independent
padded-input formula with the logical K divisor. Both variants retain every
unmasked K lane, partial M/N tiles and the complete raw-cell frame. -/
theorem run_body (variant : Bool)
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (EPS : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hinj : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN sob som son i = outOffset s N BM BN sob som son j → i = j) :
    ∃ t, stepStmts (RealSource.body Bool.false variant X W RMS OUT M N K
        sxb sxm sxk swk swn srms sob som son BM BN BK EPS) s = some t ∧
      t.pids = s.pids ∧ (∀ r o, t.undef r o = 0) ∧
      (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        t.readMem OUT (outOffset s N BM BN sob som son idx) =
          rmsSpec s X W RMS M N BM BN sxb sxm sxk swk swn srms BK (cdiv K BK) K EPS idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        o ≠ outOffset s N BM BN sob som son idx) → t.mem r o = s.mem r o) := by
  obtain ⟨s0, hpre, hP0⟩ := preLoop X W RMS s M N sxb sxm sxk swk swn srms
    BM BN BK (cdiv K BK) hundef
  obtain ⟨final, sLoop, hLoop, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := cdiv K BK) (step := 1)
      (by omega) hP0
      (fun t st hlt hinv => by
        have := rrm_step X W RMS s M N BM BN BK sxb sxm sxk swk swn srms
          (cdiv K BK) t st hlt hinv
        simpa using this)
  have hf : final = cdiv K BK := le_antisymm hPLoop.2.1 hfinal
  rw [hf] at hPLoop
  obtain ⟨t, ht, hpid, hund, hv, hframe⟩ := RealResult.postLoop X W RMS OUT s M N BM BN BK
    sxb sxm sxk swk swn srms sob som son (cdiv K BK) K EPS hinj sLoop hPLoop
  have hmem : sLoop.mem = s.mem := hPLoop.2.2.2.2.2.2.2.2.2.2.2
  refine ⟨t, ?_, hpid.trans hPLoop.1, ?_, hv, ?_⟩
  · rw [RealTransport.source_exec]
    have hpre' : stepStmts (rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms) s = some s0 := hpre
    rw [RealTransport.body, stepStmts.append_some hpre', stepStmts.cons_some hLoop, ht]
  · intro r o
    rw [hund]
    exact hPLoop.2.2.2.2.2.2.2.2.2.2.1 r o
  · intro r o h
    rw [hframe r o h, hmem]

/-- Complete callee execution at either output storage, with all original
source parameters retained. Floating erasure applies only to storage and casts. -/
theorem run (input : TypedSource.Storage) (outputFP8 variant : Bool)
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hinj : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN sob som son i = outOffset s N BM BN sob som son j → i = j) :
    ∃ t, exec (TypedSource.kernel input Bool.false outputFP8 variant X W RMS OUT M N K
        sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS).eraseDType s = some t ∧
      t.pids = s.pids ∧ (∀ r o, t.undef r o = 0) ∧
      (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        t.readMem OUT (outOffset s N BM BN sob som son idx) =
          rmsSpec s X W RMS M N BM BN sxb sxm sxk swk swn srms BK (cdiv K BK) K EPS idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        o ≠ outOffset s N BM BN sob som son idx) → t.mem r o = s.mem r o) := by
  rw [exec, ComputeKernel.toAlgKernel, RealSource.projection]
  exact run_body variant X W RMS OUT s M N K sxb sxm sxk swk swn srms sob som son BM BN BK EPS hundef hinj

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.RealExecution
