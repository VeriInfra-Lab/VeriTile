import bench.tritonbench_g.rms_rbe_matmul.Proofs.LegacyReal
import bench.tritonbench_g.rms_rbe_matmul.Proofs.Contracts
import bench.tritonbench_g.rms_rbe_matmul.Proofs.Witness
import bench.tritonbench_g.rms_rbe_matmul.Proofs.FP8Witness
import bench.tritonbench_g.rms_rbe_matmul.Proofs.Precision
import bench.tritonbench_g.rms_rbe_matmul.Proofs.NumericValues

/-! Historical mathematical and slice contracts in this file are ordinary helper
theorems. Only the retained `specification` entries below are public source contracts.
The helpers retain their statements and proofs for downstream reuse. -/


/-! Historical mathematical specifications remain available without changing their
statements. Full typed source contracts are being established in Proofs/. -/

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul
open VeriTile.Triton
set_option linter.unusedSimpArgs false

/-- **Closed-form correctness for `rms_rbe_matmul` (general statement).**

For arbitrary batch id (`s.pids 0`), linear tile pid (`s.pids 1`), tile dims
`BM`/`BN`, K-block size `BK`, and K-block count `numKBlocks` (so the
contracted dimension is `K = BK · numKBlocks`; the kernel's loads are
unmasked, so this exact-multiple presentation is required), every **active**
output cell `(i, j)` of the computed `BM × BN` tile equals

  `fp16( n_i · S[i,j] )`

over ℝ — where `S[i,j] = Σ_{t<numKBlocks} Σ_{e<BK} X[b, r(i), t·BK+e] ·
RMS[t·BK+e] · W[t·BK+e, c(j)]`,
`n_i = rsqrt((Σ_{t,e} X[b, r(i), t·BK+e]²)/K + EPS)`,
`r(i) = (pid_m·BM + i) % M`, `c(j) = (pid_n·BN + j) % N`, and every X read
carries the batch offset `b·sxb` (b = `pid_batch = s.pids 0`) — the genuine
batched RMSNorm-fused GEMM closed form of the loaded `X`/`RMS`/`W` cells, NOT
the kernel's own executed value; inactive lanes are left untouched.

Layout: `X[b,i,k]` at `X + b·sxb + offs_m(i)·sxm + k·sxk`, `W[k,j]` at
`W + k·swk + offs_n(j)·swn`, `RMS[k]` at `RMS + k·srms`, `OUT[b,i,j]` at
`OUT + b·sob + offs_m(i)·som + offs_n(j)·son`, with `pid_m = pid //
cdiv(N,BN)`, `pid_n = pid % cdiv(N,BN)` over `pid = s.pids 1`. Preconditions:
the row-major output bounds `son = 1` / `BN ≤ som` (give output-offset
injectivity; the batch term is a per-program constant) and clean initial
`undef`. -/
theorem rms_matmul_rbe_closed_form_correct
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N sxb sxm sxk swk swn srms sob som son BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hcn : son = 1) (hbnle : BN ≤ som)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_matmul_rbe_surface X W RMS OUT M N K sxb sxm sxk swk swn srms
        sob som son BM BN BK numKBlocks EPS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN)
        (fun idx => (OUT, outOffset s N BM BN sob som son idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsSpec s X W RMS M N BM BN sxb sxm sxk swk swn srms
              BK numKBlocks K EPS idx.1 idx.2.1)))) := by
  subst hcn
  have hInj : Function.Injective (outOffset s N BM BN sob som 1) := by
    have heq : outOffset s N BM BN sob som 1
        = fun idx : TileIndex [BM, BN] =>
            (s.pids 0 * sob + som * (pidM (s.pids 1) N BN * BM) + pidN (s.pids 1) N BN * BN)
              + idx.1.val * som + idx.2.1.val := by
      funext idx; simp only [outOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ som hbnle
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_matmul_rbe_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  have hmain := rrm_exec_closed_form X W RMS OUT s0 M N sxb sxm sxk swk swn srms
    sob som 1 BM BN BK numKBlocks K hK EPS hInj hundef idx
  have hExec2 : exec (rms_matmul_rbe_surface X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som 1 BM BN BK numKBlocks EPS) s0 = some s' := hExec
  rw [hExec2] at hmain
  rw [if_pos hActive] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_memcell] using hmain

/-- Both complete non-FP8 source variants have actual trace safety, successful
finite-flat execution, the independent padded-input RMS-GEMM formula and the
full raw-cell frame. K need not be divisible by BK. Bounds include all unmasked
padding reads; the output layout need be injective only on active lanes.
The six input storage labels are a formal family: pinned native dot admission,
IEEE rounding, the FP8 value branch and the separate rotary JIT remain outside
this exact-real statement. -/
specification half_formal_real_correct (input : TypedSource.Storage) (variant : Bool)
    (A : FlatAlloc) (s : BlockState) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (hu : s.undef = fun _ _ => 0) (hcol : 0 < son) (hrow : N * son ≤ som)
    (hb : AccessSafety.Reads A.extent s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < A.extent OUT) :
    VeriTile.Spec.Real (Contracts.ExactIO
      (TypedSource.kernel input Bool.false variant X W RMS OUT M N K sxb sxm sxk swk swn srms
        sob som son BM BN BK start rbe theta EPS)
      A s X W RMS OUT M N K sxb sxm sxk swk swn srms sob som son BM BN BK EPS) := by
  refine ⟨?_, FlatIO.half input variant A X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS hd hc hu
    (Geometry.active_injective s M N BM BN sob som son hcol hrow) hb ho⟩
  exact AccessSafety.complete_half input variant A.extent X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS
    (fun r o => by rw [hu]) hb ho

/-- Both complete FP8-weight source variants have actual trace safety,
successful finite-flat execution and the independent padded-array RMS-GEMM
formula with a complete raw-cell frame. All weight premises demand actual
integer bytes with finite E5M2 decodes; the output is the real interpretation
of an E5M2 floating carrier, not a packed-byte encoding theorem. The six X
labels are a formal family; native mixed-dot admission, physical IEEE rounding
and the separate rotary JIT remain outside this exact-real statement. -/
specification fp8_formal_real_correct (input : TypedSource.Storage) (variant : Bool)
    (A : FlatAlloc) (s : BlockState) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (cdiv K BK) → Fin BK → ByteExecution.Byte)
    (ws : Fin (cdiv K BK) → Fin BK → Fin BN → ByteExecution.Byte)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (hu : s.undef = fun _ _ => 0) (hcol : 0 < son) (hrow : N * son ≤ som)
    (hXW : X ≠ W) (hXR : X ≠ RMS)
    (hinput : ByteExecution.Inputs s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms xs rs ws)
    (hb : AccessSafety.Reads A.extent s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < A.extent OUT) :
    VeriTile.Spec.Real (Contracts.ByteIO
      (TypedSource.kernel input Bool.true variant X W RMS OUT M N K sxb sxm sxk swk swn srms
        sob som son BM BN BK start rbe theta EPS)
      A s OUT M N K sob som son BM BN BK EPS xs rs ws) := by
  refine ⟨?_, FlatIO.bytes input variant A X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS xs rs ws
    hd hc hu hcol hrow hXW hXR hinput hb ho⟩
  exact ByteExecution.complete input variant A.extent X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS
    (fun r o => by rw [hu]) hXW hXR
    (ByteExecution.inputs_bytes s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms xs rs ws hinput)
    hb ho

end VeriTile.Bench.TritonBenchG.RmsRbeMatmul
