import bench.tritonbench_g.llama_ff_triton.Proofs.Transport
import bench.tritonbench_g.llama_ff_triton.Proofs.Result
import bench.tritonbench_g.llama_ff_triton.Proofs.Geometry
import bench.tritonbench_g.llama_ff_triton.Proofs.Math

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.RealExecution
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- Successful complete fp16 execution, including every ceiling-counted K
block, all active outputs and preservation of all other typed cells. -/
theorem run (variant : Bool) (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hinj : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN soutm soutn i = outOffset s N BM BN soutm soutn j → i = j) :
    ∃ t, exec (TypedSource.kernel Bool.false variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).eraseDType s = some t ∧
      (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        t.readMem OUT (outOffset s N BM BN soutm soutn idx) =
          ffSpec s A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms BK (cdiv K BK) K EPS idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        o ≠ outOffset s N BM BN soutm soutn idx) → t.mem r o = s.mem r o) := by
  obtain ⟨s0, hpre, hP0⟩ := preLoop A W1 W3 OUT RMS s M N sam sak sw1k sw1n sw3k sw3n
    soutm soutn srms BM BN BK (cdiv K BK) K EPS hundef
  obtain ⟨final, sLoop, hLoop, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "_i") (start := 0) (stop := cdiv K BK) (step := 1)
      (by omega) hP0
      (fun t st hlt hinv => by
        have := ff_step A W1 W3 RMS s M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms
          (cdiv K BK) t st hlt hinv
        simpa using this)
  have hf : final = cdiv K BK := le_antisymm hPLoop.2.1 hfinal
  rw [hf] at hPLoop
  obtain ⟨t, ht, hv, hframe⟩ := RealResult.postLoop A W1 W3 OUT RMS s M N BM BN BK
    sam sak sw1k sw1n sw3k sw3n srms soutm soutn (cdiv K BK) K EPS hinj sLoop hPLoop
  have hmem : sLoop.mem = s.mem := hPLoop.2.2.2.2.2.2.2.2.2.2.2.2
  refine ⟨t, ?_, hv, ?_⟩
  · rw [exec, ComputeKernel.toAlgKernel, RealSource.projection]
    change stepStmts (RealSource.body Bool.false variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS) s = some t
    rw [RealTransport.source_exec]
    have hpre' : stepStmts (ffPrologue A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms) s = some s0 := hpre
    rw [RealTransport.body, stepStmts.append_some hpre', stepStmts.cons_some hLoop, ht]
  · intro r o h
    rw [hframe r o h, hmem]

/-- Input-array relation for all actual source reads, including wrapped M/N
lanes and the unmasked final K block. This has no output/execution premise. -/
def Inputs (s : BlockState) (A W1 W3 RMS : RegionName)
    (M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms : Nat)
    (xs : Fin (Mathematics.steps K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (Mathematics.steps K BK) → Fin BK → ℝ)
    (w1 w3 : Fin (Mathematics.steps K BK) → Fin BK → Fin BN → ℝ) : Prop :=
  (∀ t i e, aElem s A M N BM BN sam sak i (t.val * BK + e.val) = xs t i e) ∧
  (∀ t e, rmsElem s RMS srms (t.val * BK + e.val) = rs t e) ∧
  (∀ t e j, wElem s W1 N BN sw1k sw1n j (t.val * BK + e.val) = w1 t e j) ∧
  (∀ t e j, wElem s W3 N BN sw3k sw3n j (t.val * BK + e.val) = w3 t e j)

theorem formula (s : BlockState) (A W1 W3 RMS : RegionName)
    (M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms : Nat) (EPS : ℝ)
    (xs : Fin (Mathematics.steps K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (Mathematics.steps K BK) → Fin BK → ℝ)
    (w1 w3 : Fin (Mathematics.steps K BK) → Fin BK → Fin BN → ℝ)
    (h : Inputs s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms xs rs w1 w3)
    (i : Fin BM) (j : Fin BN) :
    ffSpec s A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms BK (cdiv K BK) K EPS i j =
      Mathematics.output K BK BM BN EPS xs rs w1 w3 i j := by
  rcases h with ⟨ha, hr, h1, h3⟩
  simp only [ffSpec, normSpec, accPartial, Mathematics.output, Mathematics.norm,
    Mathematics.weightedSum, PaddedRMSGemm.norm, PaddedRMSGemm.weightedSum, TiledActivation.silu, one_div]
  simp only [Mathematics.steps] at ha hr h1 h3 ⊢
  simp only [ha, hr, h1, h3]
  rfl

/-- Both source variants realize the independent SwiGLU array formula. The
only layout condition concerns active outputs; partial row/column tiles and
nondivisible K are retained. -/
theorem arrays (variant : Bool) (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ)
    (xs : Fin (Mathematics.steps K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (Mathematics.steps K BK) → Fin BK → ℝ)
    (w1 w3 : Fin (Mathematics.steps K BK) → Fin BK → Fin BN → ℝ)
    (hu : ∀ rg o, s.undef rg o = 0) (hc : 0 < soutn) (hr : N * soutn ≤ soutm)
    (hinput : Inputs s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms xs rs w1 w3) :
    ∃ t, exec (TypedSource.kernel Bool.false variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).eraseDType s = some t ∧
      (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        t.readMem OUT (outOffset s N BM BN soutm soutn idx) =
          Mathematics.output K BK BM BN EPS xs rs w1 w3 idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        o ≠ outOffset s N BM BN soutm soutn idx) → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := run variant A W1 W3 OUT RMS s M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS hu
    (Geometry.active_injective s M N BM BN soutm soutn hc hr)
  refine ⟨t, he, ?_, hf⟩
  intro idx hi
  rw [hv idx hi, formula s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms EPS xs rs w1 w3 hinput]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.RealExecution
