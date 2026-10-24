import bench.tritonbench_g.llama_ff_triton.Proofs.LegacyReal
import bench.tritonbench_g.llama_ff_triton.Proofs.Contracts
import bench.tritonbench_g.llama_ff_triton.Proofs.Witness
import bench.tritonbench_g.llama_ff_triton.Proofs.ExecutionValues
import bench.tritonbench_g.llama_ff_triton.Proofs.Precision
import VeriTile.Meta.StatementAudit

/-! Historical mathematical and slice contracts in this file are ordinary helper
theorems. Only the retained `specification` entries below are public source contracts.
The helpers retain their statements and proofs for downstream reuse. -/


namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton
open VeriTile.Triton

/-- **Closed-form correctness for `llama_ff_triton` (general statement).**

For arbitrary linear program id `pid`, tile dims `BM`/`BN`, K-block size `BK`,
and K-block count `numKBlocks` (so the contracted dimension is
`K = BK · numKBlocks`; the kernel's loads are unmasked, so this exact-multiple
presentation is required), every **active** output cell `(i, j)` of the
computed `BM × BN` tile equals

  `fp16( silu(n_i · S1[i,j]) * (n_i · S2[i,j]) )`

over ℝ — where `S1[i,j] = Σ_{t<numKBlocks} Σ_{e<BK} A[r(i), t·BK+e] ·
RMS[t·BK+e] · W1[t·BK+e, c(j)]` (similarly `S2` with `W3`),
`n_i = rsqrt((Σ_{t,e} A[r(i), t·BK+e]²)/K + EPS)`, `silu x = x · sigmoid x`,
`r(i) = (pid_m·BM + i) % M`, `c(j) = (pid_n·BN + j) % N` — the genuine
RMSNorm-fused SwiGLU closed form of the loaded `A`/`RMS`/`W1`/`W3` cells, NOT
the kernel's own executed value; inactive lanes are left untouched.

Layout: `A[i,k]` at `A + offs_am(i)·sam + k·sak`, `W[k,j]` at
`W + k·swk + offs_bn(j)·swn`, `RMS[k]` at `RMS + k·srms`, `OUT[i,j]` at
`OUT + soutm·offs_outm(i) + soutn·offs_outn(j)`, with `pid_m = pid //
cdiv(N,BN)`, `pid_n = pid % cdiv(N,BN)`. Preconditions: the row-major output
bounds `soutn = 1` / `BN ≤ soutm` (give output-offset injectivity) and clean
initial `undef`. -/
theorem ff_llama_closed_form_correct
    (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hcn : soutn = 1) (hbnle : BN ≤ soutm)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := ff_llama_surface A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n
        soutm soutn srms BM BN BK numKBlocks EPS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN)
        (fun idx => (OUT, outOffset s N BM BN soutm soutn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (ffSpec s A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms
              BK numKBlocks K EPS idx.1 idx.2.1)))) := by
  subst hcn
  have hInj : Function.Injective (outOffset s N BM BN soutm 1) := by
    have heq : outOffset s N BM BN soutm 1
        = fun idx : TileIndex [BM, BN] =>
            (soutm * (pidM (s.pids 0) N BN * BM) + pidN (s.pids 0) N BN * BN)
              + idx.1.val * soutm + idx.2.1.val := by
      funext idx; simp only [outOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ soutm hbnle
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [ff_llama_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  have hmain := ff_llama_exec_closed_form A W1 W3 OUT RMS s0 M N sam sak sw1k sw1n sw3k sw3n
    soutm 1 srms BM BN BK numKBlocks K hK EPS hInj hundef idx
  have hExec2 : exec (ff_llama_surface A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n
      soutm 1 srms BM BN BK numKBlocks EPS) s0 = some s' := hExec
  rw [hExec2] at hmain
  rw [if_pos hActive] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_memcell] using hmain


/-- Complete fp16-weight source execution over finite flat storage, with the
independent padded-array formula and the full frame of typed memory cells. -/
specification fp16_flat_correct (variant : Bool) (alloc : FlatAlloc) (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ)
    (xs : Fin (Mathematics.steps K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (Mathematics.steps K BK) → Fin BK → ℝ)
    (w1 w3 : Fin (Mathematics.steps K BK) → Fin BK → Fin BN → ℝ)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hout : OUT ∈ alloc.regions)
    (hu : ∀ rg o, s.undef rg o = 0) (hc : 0 < soutn) (hr : N * soutn ≤ soutm)
    (hinput : RealExecution.Inputs s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms xs rs w1 w3)
    (hb : AccessSafety.Reads alloc.extent s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN soutm soutn i < alloc.extent OUT) :
    Spec.Real (Contracts.ExactIO
      (TypedSource.kernel Bool.false variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n
        sw3k sw3n soutm soutn srms BM BN BK EPS)
      alloc s OUT M N K soutm soutn BM BN BK EPS xs rs w1 w3) := by
  exact ⟨AccessSafety.complete_fp16 variant alloc.extent A W1 W3 OUT RMS s M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS hu hb ho,
    FlatIO.half variant alloc A W1 W3 OUT RMS s M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS xs rs w1 w3 hd hcov hout hu hc hr hinput hb ho⟩

/-- Complete signed-byte E5M2 source execution over finite flat storage, with the
independent padded-array formula and the full frame of typed memory cells. -/
specification fp8_flat_correct (variant : Bool) (alloc : FlatAlloc) (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ)
    (xs : Fin (Mathematics.steps K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (Mathematics.steps K BK) → Fin BK → Mathematics.Byte)
    (w1 w3 : Fin (Mathematics.steps K BK) → Fin BK → Fin BN → Mathematics.Byte)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hout : OUT ∈ alloc.regions)
    (hu : ∀ rg o, s.undef rg o = 0) (hc : 0 < soutn) (hr : N * soutn ≤ soutm)
    (hA1 : A ≠ W1) (hA3 : A ≠ W3) (hAR : A ≠ RMS)
    (hinput : ByteExecution.Inputs s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms xs rs w1 w3)
    (hb : AccessSafety.Reads alloc.extent s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN soutm soutn i < alloc.extent OUT) :
    Spec.Real (Contracts.ByteIO
      (TypedSource.kernel Bool.true variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n
        sw3k sw3n soutm soutn srms BM BN BK EPS)
      alloc s OUT M N K soutm soutn BM BN BK EPS xs rs w1 w3) := by
  exact ⟨ByteExecution.complete variant alloc.extent A W1 W3 OUT RMS s M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS hu hA1 hA3 hAR
    (ByteExecution.inputs_bytes s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms xs rs w1 w3 hinput) hb ho,
    FlatIO.bytes variant alloc A W1 W3 OUT RMS s M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS xs rs w1 w3 hd hcov hout hu hc hr hA1 hA3 hAR hinput hb ho⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton
