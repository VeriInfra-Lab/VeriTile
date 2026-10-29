import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKVByteWitness
import bench.tritonbench_g.rms_matmul_rbe.Proofs.NumericExecution
import bench.tritonbench_g.rms_matmul_rbe.Proofs.LegacyReal
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Contracts
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Precision
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Witness
import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKVWitness

/-! Historical mathematical and slice contracts in this file are ordinary helper
theorems. Only the retained `specification` entries below are public source contracts.
The helpers retain their statements and proofs for downstream reuse. -/


namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe
open VeriTile.Triton
set_option linter.unusedSimpArgs false

/-- **Closed-form correctness for `rms_matmul_rbe` (general statement).**

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
  obtain ⟨sP, hP, _hpids, _hund, _hfr, hstore⟩ := rrm_pass X W RMS OUT s0 M N sxb sxm sxk
    swk swn srms sob som 1 BM BN BK numKBlocks K hK EPS hInj hundef
  have hExec2 : exec (rms_matmul_rbe_surface X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som 1 BM BN BK numKBlocks EPS) s0 = some s' := hExec
  have hexecP : exec (rms_matmul_rbe_surface X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som 1 BM BN BK numKBlocks EPS) s0 = some sP := by
    rw [exec, rrm_body_split X W RMS OUT M N K sxb sxm sxk swk swn srms sob som 1
      BM BN BK numKBlocks EPS]
    exact hP
  have hs' : s' = sP := by
    rw [hExec2] at hexecP
    exact Option.some.inj hexecP
  subst hs'
  have hmain := hstore idx
  rw [if_pos hActive] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_memcell] using hmain

set_option maxHeartbeats 4000000 in
/-- **Closed-form correctness for `rms_matmul_rbe_qkv` (general statement).**

The qkv kernel is the callee `rms_matmul_rbe` inlined three times — at
`(q_weight, q)`, `(k_weight, k)`, `(v_weight, v)` — all sharing `X`, `RMS`,
and the launch geometry. The conclusion is the standard multi-output bundle:
a conjunction of three `Realizes_without_Rounding`, one per stored output.
For arbitrary batch id (`s.pids 0`) and linear tile pid (`s.pids 1`), every
**active** output cell `(i, j)` of the computed `BM × BN` tile of each
output equals

  `fp16( n_i · S_w[i,j] )`

over ℝ — where `S_w[i,j] = Σ_{t<numKBlocks} Σ_{e<BK} X[b, r(i), t·BK+e] ·
RMS[t·BK+e] · W[t·BK+e, c(j)]` against the pass's OWN weight matrix
(`W = QW`/`KW`/`VW` with its own strides) and
`n_i = rsqrt((Σ_{t,e} X[b, r(i), t·BK+e]²)/K + EPS)` is the SHARED RMS
normalizer; `r(i) = (pid_m·BM + i) % M`, `c(j) = (pid_n·BN + j) % N`, batch
offset `b·sxb` in every X read — the genuine closed form of the loaded
`X`/`RMS`/weight cells, NOT the kernel's own executed value; inactive lanes
impose no obligation.

Preconditions: `K = BK · numKBlocks` (unmasked loads); per-output row-major
bounds `sqn = 1`/`BN ≤ sqm`, `skn = 1`/`BN ≤ skm`, `svn = 1`/`BN ≤ svm`
(output-offset injectivity); pairwise distinctness of the written regions
`Q`/`KOut`/`V` from each other and from the read regions each later pass
consumes (`X`, `RMS`, and the pass's weight region — the host passes eight
distinct buffers), which is exactly what the cross-pass frame transport
needs; and clean initial `undef`. -/
theorem rms_matmul_rbe_qkv_closed_form_correct
    (X QW KW VW RMS Q KOut V : RegionName) (s : BlockState)
    (M N sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hqn : sqn = 1) (hqm : BN ≤ sqm)
    (hkn : skn = 1) (hkm : BN ≤ skm)
    (hvn : svn = 1) (hvm : BN ≤ svm)
    (hXQ : X ≠ Q) (hRMSQ : RMS ≠ Q) (hKWQ : KW ≠ Q) (hVWQ : VW ≠ Q)
    (hXK : X ≠ KOut) (hRMSK : RMS ≠ KOut) (hVWK : VW ≠ KOut)
    (hQK : Q ≠ KOut) (hQV : Q ≠ V) (hKV : KOut ≠ V)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
        BM BN BK numKBlocks EPS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN)
        (fun idx => (Q, outOffset s N BM BN sqb sqm sqn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsSpec s X QW RMS M N BM BN sxb sxm sxk sqwk sqwn srms
              BK numKBlocks K EPS idx.1 idx.2.1))))
    ∧ ComputeCorrect.Realizes_without_Rounding
        (kernel := rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
          sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
          BM BN BK numKBlocks EPS)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (active s M N BM BN)
          (fun idx => (KOut, outOffset s N BM BN skb skm skn idx)))
        (expected := fun idx : TileIndex [BM, BN] =>
          MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (rmsSpec s X KW RMS M N BM BN sxb sxm sxk skwk skwn srms
                BK numKBlocks K EPS idx.1 idx.2.1))))
    ∧ ComputeCorrect.Realizes_without_Rounding
        (kernel := rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
          sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
          BM BN BK numKBlocks EPS)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (active s M N BM BN)
          (fun idx => (V, outOffset s N BM BN svb svm svn idx)))
        (expected := fun idx : TileIndex [BM, BN] =>
          MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (rmsSpec s X VW RMS M N BM BN sxb sxm sxk svwk svwn srms
                BK numKBlocks K EPS idx.1 idx.2.1)))) := by
  subst hqn
  subst hkn
  subst hvn
  have hInjQ : Function.Injective (outOffset s N BM BN sqb sqm 1) := by
    have heq : outOffset s N BM BN sqb sqm 1
        = fun idx : TileIndex [BM, BN] =>
            (s.pids 0 * sqb + sqm * (pidM (s.pids 1) N BN * BM) + pidN (s.pids 1) N BN * BN)
              + idx.1.val * sqm + idx.2.1.val := by
      funext idx; simp only [outOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ sqm hqm
  have hInjK : Function.Injective (outOffset s N BM BN skb skm 1) := by
    have heq : outOffset s N BM BN skb skm 1
        = fun idx : TileIndex [BM, BN] =>
            (s.pids 0 * skb + skm * (pidM (s.pids 1) N BN * BM) + pidN (s.pids 1) N BN * BN)
              + idx.1.val * skm + idx.2.1.val := by
      funext idx; simp only [outOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ skm hkm
  have hInjV : Function.Injective (outOffset s N BM BN svb svm 1) := by
    have heq : outOffset s N BM BN svb svm 1
        = fun idx : TileIndex [BM, BN] =>
            (s.pids 0 * svb + svm * (pidM (s.pids 1) N BN * BM) + pidN (s.pids 1) N BN * BN)
              + idx.1.val * svm + idx.2.1.val := by
      funext idx; simp only [outOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ svm hvm
  refine ⟨?_, ?_, ?_⟩
  · -- Q realization
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [rms_matmul_rbe_qkv_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst hs0
    intro idx hActive
    have hmain := qkv_exec_closed_form X QW KW VW RMS Q KOut V s0 M N sxb sxm sxk
      sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
      BM BN BK numKBlocks K hK EPS hInjQ hInjK hInjV
      hXQ hRMSQ hKWQ hVWQ hXK hRMSK hVWK hQK hQV hKV hundef
    have hExec2 : exec (rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
        BM BN BK numKBlocks EPS) s0 = some s' := hExec
    rw [hExec2] at hmain
    have hcell := hmain.1 idx hActive
    simpa only [ComputeCorrect.OutputReadable.read_memcell] using hcell
  · -- K realization
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [rms_matmul_rbe_qkv_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst hs0
    intro idx hActive
    have hmain := qkv_exec_closed_form X QW KW VW RMS Q KOut V s0 M N sxb sxm sxk
      sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
      BM BN BK numKBlocks K hK EPS hInjQ hInjK hInjV
      hXQ hRMSQ hKWQ hVWQ hXK hRMSK hVWK hQK hQV hKV hundef
    have hExec2 : exec (rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
        BM BN BK numKBlocks EPS) s0 = some s' := hExec
    rw [hExec2] at hmain
    have hcell := hmain.2.1 idx hActive
    simpa only [ComputeCorrect.OutputReadable.read_memcell] using hcell
  · -- V realization
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [rms_matmul_rbe_qkv_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst hs0
    intro idx hActive
    have hmain := qkv_exec_closed_form X QW KW VW RMS Q KOut V s0 M N sxb sxm sxk
      sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
      BM BN BK numKBlocks K hK EPS hInjQ hInjK hInjV
      hXQ hRMSQ hKWQ hVWQ hXK hRMSK hVWK hQK hQV hKV hundef
    have hExec2 : exec (rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm 1 skb skm 1 svb svm 1
        BM BN BK numKBlocks EPS) s0 = some s' := hExec
    rw [hExec2] at hmain
    have hcell := hmain.2.2 idx hActive
    simpa only [ComputeCorrect.OutputReadable.read_memcell] using hcell
/-- Both complete non-FP8 source variants have actual trace safety, successful
finite-flat execution, the independent padded-input RMS-GEMM formula and the
full raw-cell frame. K need not be divisible by BK. Bounds include all unmasked
padding reads; the output layout need be injective only on active lanes.
The six input storage labels are a formal family: pinned native dot admission,
IEEE rounding, the FP8 value branch remain outside
this exact-real statement. -/
specification half_formal_real_correct (input : TypedSource.Storage) (outputFP8 variant : Bool)
    (A : FlatAlloc) (s : BlockState) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (hu : s.undef = fun _ _ => 0) (hcol : 0 < son) (hrow : N * son ≤ som)
    (hb : AccessSafety.Reads A.extent s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < A.extent OUT) :
    VeriTile.Spec.Real (Contracts.ExactIO
      (TypedSource.kernel input Bool.false outputFP8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms
        sob som son BM BN BK start rbe theta EPS)
      A s X W RMS OUT M N K sxb sxm sxk swk swn srms sob som son BM BN BK EPS) := by
  refine ⟨?_, FlatIO.half input outputFP8 variant A X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS hd hc hu
    (Geometry.active_injective s M N BM BN sob som son hcol hrow) hb ho⟩
  exact AccessSafety.complete_half input outputFP8 variant A.extent X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS
    (fun r o => by rw [hu]) hb ho

/-- Complete sequential QKV real IO for both variants and all formal
floating X storage labels. Each output uses its own initial weight matrix and
stride tuple; all unmasked padding reads require finite bounds. Native/IEEE
admission and the byte-value real theorem are separate obligations. -/
specification qkv_half_formal_real_correct (input : TypedSource.Storage) (variant : Bool) (A : FlatAlloc)
    (X QW KW VW RMS Q KOut V : RegionName) (s : BlockState)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ)
    (hu : s.undef = fun _ _ => 0)
    (hiQ : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN sqb sqm sqn i = outOffset s N BM BN sqb sqm sqn j → i = j)
    (hiK : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN skb skm skn i = outOffset s N BM BN skb skm skn j → i = j)
    (hiV : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN svb svm svn i = outOffset s N BM BN svb svm svn j → i = j)
    (hXQ : X ≠ Q) (hRQ : RMS ≠ Q) (hKWQ : KW ≠ Q) (hVWQ : VW ≠ Q)
    (hXK : X ≠ KOut) (hRK : RMS ≠ KOut) (hVWK : VW ≠ KOut)
    (hQK : Q ≠ KOut) (hQV : Q ≠ V) (hKV : KOut ≠ V)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (bq : AccessSafety.Reads A.extent s X QW RMS M N K BM BN BK sxb sxm sxk sqwk sqwn srms)
    (bk : AccessSafety.Reads A.extent s X KW RMS M N K BM BN BK sxb sxm sxk skwk skwn srms)
    (bv : AccessSafety.Reads A.extent s X VW RMS M N K BM BN BK sxb sxm sxk svwk svwn srms)
    (oq : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN sqb sqm sqn i < A.extent Q)
    (ok : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN skb skm skn i < A.extent KOut)
    (ov : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN svb svm svn i < A.extent V) :
    VeriTile.Spec.Real (Contracts.QKVExactIO
      (TypedSource.qkv input Bool.false variant X QW KW VW RMS Q KOut V M N K
        sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS)
      A s X QW KW VW RMS Q KOut V M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK EPS) := by
  refine ⟨?_, QKVFlat.half input variant A X QW KW VW RMS Q KOut V s M N K
    sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn BM BN BK
    start theta EPS hu hiQ hiK hiV hXQ hRQ hKWQ hVWQ hXK hRK hVWK hQK hQV hKV
    hd hc bq bk bv oq ok ov⟩
  exact QKVFlat.safe_half input variant A.extent X QW KW VW RMS Q KOut V s M N K
    sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn BM BN BK
    start theta EPS (fun r o => by rw [hu]) hiQ hiK hiV bq bk bv oq ok ov

/-- Both complete FP8-weight source variants have actual trace safety,
successful finite-flat execution and the independent padded-array RMS-GEMM
formula with a complete raw-cell frame. All weight premises demand actual
integer bytes with finite E5M2 decodes; the output is the real interpretation
of the selected fp16 or E5M2 carrier, not a packed-byte encoding theorem. The
six X labels are a formal family; native mixed-dot admission and physical IEEE
rounding remain outside this exact-real statement. -/
specification fp8_formal_real_correct (input : TypedSource.Storage) (outputFP8 variant : Bool)
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
      (TypedSource.kernel input Bool.true outputFP8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms
        sob som son BM BN BK start rbe theta EPS)
      A s OUT M N K sob som son BM BN BK EPS xs rs ws) := by
  refine ⟨?_, FlatIO.bytes input outputFP8 variant A X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS xs rs ws
    hd hc hu hcol hrow hXW hXR hinput hb ho⟩
  exact ByteExecution.complete input outputFP8 variant A.extent X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS
    (fun r o => by rw [hu]) hXW hXR
    (ByteExecution.inputs_bytes s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms xs rs ws hinput)
    hb ho


/-- Full finite-flat QKV byte IO from independent finite decoded arrays.
Q retains fp16 storage; K/V use E5M2. Every padded read and every unwritten
raw cell is covered. This theorem uses exact real arithmetic, not IEEE rounding. -/
specification qkv_fp8_formal_real_correct (input : TypedSource.Storage) (variant : Bool) (A : FlatAlloc)
    (X QW KW VW RMS Q KOut V : RegionName) (s : BlockState)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ)
    (xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (cdiv K BK) → Fin BK → ByteExecution.Byte)
    (qws kws vws : Fin (cdiv K BK) → Fin BK → Fin BN → ByteExecution.Byte)
    (hu : s.undef = fun _ _ => 0)
    (hiQ : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN sqb sqm sqn i = outOffset s N BM BN sqb sqm sqn j → i = j)
    (hiK : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN skb skm skn i = outOffset s N BM BN skb skm skn j → i = j)
    (hiV : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN svb svm svn i = outOffset s N BM BN svb svm svn j → i = j)
    (hXQW : X ≠ QW) (hXKW : X ≠ KW) (hXVW : X ≠ VW) (hXR : X ≠ RMS)
    (inputQ : ByteExecution.Inputs s X QW RMS M N K BM BN BK sxb sxm sxk sqwk sqwn srms xs rs qws)
    (inputK : ByteExecution.Inputs s X KW RMS M N K BM BN BK sxb sxm sxk skwk skwn srms xs rs kws)
    (inputV : ByteExecution.Inputs s X VW RMS M N K BM BN BK sxb sxm sxk svwk svwn srms xs rs vws)
    (hXQ : X ≠ Q) (hRQ : RMS ≠ Q) (hKWQ : KW ≠ Q) (hVWQ : VW ≠ Q)
    (hXK : X ≠ KOut) (hRK : RMS ≠ KOut) (hVWK : VW ≠ KOut)
    (hQK : Q ≠ KOut) (hQV : Q ≠ V) (hKV : KOut ≠ V)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (bq : AccessSafety.Reads A.extent s X QW RMS M N K BM BN BK sxb sxm sxk sqwk sqwn srms)
    (bk : AccessSafety.Reads A.extent s X KW RMS M N K BM BN BK sxb sxm sxk skwk skwn srms)
    (bv : AccessSafety.Reads A.extent s X VW RMS M N K BM BN BK sxb sxm sxk svwk svwn srms)
    (oq : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN sqb sqm sqn i < A.extent Q)
    (ok : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN skb skm skn i < A.extent KOut)
    (ov : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN svb svm svn i < A.extent V) :
    VeriTile.Spec.Real (Contracts.QKVByteIO
      (TypedSource.qkv input Bool.true variant X QW KW VW RMS Q KOut V M N K
        sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS)
      A s Q KOut V M N K sqb sqm sqn skb skm skn svb svm svn BM BN BK EPS xs rs qws kws vws) := by
  refine ⟨?_, QKVByteFlat.arrays input variant A X QW KW VW RMS Q KOut V s M N K
    sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn BM BN BK
    start theta EPS xs rs qws kws vws hu hiQ hiK hiV hXQW hXKW hXVW hXR inputQ inputK inputV
    hXQ hRQ hKWQ hVWQ hXK hRK hVWK hQK hQV hKV hd hc bq bk bv oq ok ov⟩
  exact QKVByteFlat.safe_bytes input variant A.extent X QW KW VW RMS Q KOut V s M N K
    sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn BM BN BK
    start theta EPS (fun r o => by rw [hu]) hiQ hiK hiV hXQW hXKW hXVW hXR
    (ByteExecution.inputs_bytes s X QW RMS M N K BM BN BK sxb sxm sxk sqwk sqwn srms xs rs qws inputQ)
    (ByteExecution.inputs_bytes s X KW RMS M N K BM BN BK sxb sxm sxk skwk skwn srms xs rs kws inputK)
    (ByteExecution.inputs_bytes s X VW RMS M N K BM BN BK sxb sxm sxk svwk svwn srms xs rs vws inputV)
    hXQ hRQ hKWQ hVWQ hXK hRK hVWK hQK hQV hKV bq bk bv oq ok ov

end VeriTile.Bench.TritonBenchG.RmsMatmulRbe
