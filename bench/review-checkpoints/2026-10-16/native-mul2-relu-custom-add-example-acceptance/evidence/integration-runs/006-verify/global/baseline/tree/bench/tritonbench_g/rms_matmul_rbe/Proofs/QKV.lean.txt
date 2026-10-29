import bench.tritonbench_g.rms_matmul_rbe.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.QKVExecution
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Projection commutes with concatenation of the actual compute statements.
A rejected stage cannot be turned into an empty successful computation. -/
theorem erased_append (ins outs : List RegionName) (a b : ComputeKernel)
    (ka kb : Kernel)
    (ha : a.eraseDType.toAlgorithm? = Except.ok ka)
    (hb : b.eraseDType.toAlgorithm? = Except.ok kb) :
    (ComputeKernel.mk ins outs (a.surfaceBody ++ b.surfaceBody)).eraseDType.toAlgorithm? =
      Except.ok ⟨ins, outs, ka.body ++ kb.body⟩ := by
  cases a with | mk ia oa ba =>
    cases b with | mk ib ob bb =>
      cases hpa : ComputeStmt.listToAlgorithm? ba with
      | error e => simp [ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, hpa, Except.bind, bind] at ha
      | ok pa =>
        cases hpb : ComputeStmt.listToAlgorithm? bb with
        | error e => simp [ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, hpb, Except.bind, bind] at hb
        | ok pb =>
          simp only [ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, hpa, hpb,
            Except.bind, bind, Kernel.eraseDType, ComputeKernel.fromKernelBody,
            ComputeStmt.listToAlgorithm?_map_alg, Except.ok.injEq] at ha hb
          subst ka
          subst kb
          simp [ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
            ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?_append, hpa, hpb,
            Kernel.eraseDType, Stmt.eraseDTypeList, ComputeKernel.fromKernelBody,
            ComputeStmt.listToAlgorithm?_map_alg, List.map_append, Except.bind, bind]
          clear hpa
          induction pa with
          | nil => simp [Stmt.eraseDTypeList]
          | cons st rest ih => simp [Stmt.eraseDTypeList, ih]

def body (fp8 variant : Bool) (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat) (EPS : ℝ) : List Stmt :=
  RealSource.body fp8 variant X QW RMS Q M N K sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK EPS ++
  RealSource.body fp8 variant X KW RMS KOut M N K sxb sxm sxk skwk skwn srms skb skm skn BM BN BK EPS ++
  RealSource.body fp8 variant X VW RMS V M N K sxb sxm sxk svwk svwn srms svb svm svn BM BN BK EPS

/-- All three source passes project, including the independent Q versus K/V
output storage and the unchanged unused rotary flags. -/
theorem projection (input : TypedSource.Storage) (fp8 variant : Bool)
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ) :
    (TypedSource.qkv input fp8 variant X QW KW VW RMS Q KOut V M N K
      sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS).eraseDType.toAlgorithm? =
      Except.ok ⟨[X, QW, KW, VW, RMS], [Q, KOut, V],
        body fp8 variant X QW KW VW RMS Q KOut V M N K
          sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
          sqb sqm sqn skb skm skn svb svm svn BM BN BK EPS⟩ := by
  let a := TypedSource.kernel input fp8 Bool.false variant X QW RMS Q M N K
    sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK start Bool.true theta EPS
  let b := TypedSource.kernel input fp8 fp8 variant X KW RMS KOut M N K
    sxb sxm sxk skwk skwn srms skb skm skn BM BN BK start Bool.true theta EPS
  let c := TypedSource.kernel input fp8 fp8 variant X VW RMS V M N K
    sxb sxm sxk svwk svwn srms svb svm svn BM BN BK start Bool.false theta EPS
  have hab := erased_append [] [] a b _ _
    (RealSource.projection input fp8 Bool.false variant X QW RMS Q M N K
      sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK start Bool.true theta EPS)
    (RealSource.projection input fp8 fp8 variant X KW RMS KOut M N K
      sxb sxm sxk skwk skwn srms skb skm skn BM BN BK start Bool.true theta EPS)
  exact erased_append [X, QW, KW, VW, RMS] [Q, KOut, V]
    (.mk [] [] (a.surfaceBody ++ b.surfaceBody)) c _ _ hab
    (RealSource.projection input fp8 fp8 variant X VW RMS V M N K
      sxb sxm sxk svwk svwn srms svb svm svn BM BN BK start Bool.false theta EPS)

/-- The complete sequence produces Q, K and V from their own initial weight
regions and independent strides, and frames every cell outside all three
active footprints. The actual ceil-K iteration count is retained. -/
theorem run (input : TypedSource.Storage) (variant : Bool)
    (X QW KW VW RMS Q KOut V : RegionName) (s : BlockState)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ)
    (hu : ∀ r o, s.undef r o = 0)
    (hiQ : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN sqb sqm sqn i = outOffset s N BM BN sqb sqm sqn j → i = j)
    (hiK : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN skb skm skn i = outOffset s N BM BN skb skm skn j → i = j)
    (hiV : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN svb svm svn i = outOffset s N BM BN svb svm svn j → i = j)
    (hXQ : X ≠ Q) (hRQ : RMS ≠ Q) (hKWQ : KW ≠ Q) (hVWQ : VW ≠ Q)
    (hXK : X ≠ KOut) (hRK : RMS ≠ KOut) (hVWK : VW ≠ KOut)
    (hQK : Q ≠ KOut) (hQV : Q ≠ V) (hKV : KOut ≠ V) :
    ∃ t, exec (TypedSource.qkv input Bool.false variant X QW KW VW RMS Q KOut V M N K
        sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS).eraseDType s = some t ∧
      t.pids = s.pids ∧ (∀ r o, t.undef r o = 0) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem Q (outOffset s N BM BN sqb sqm sqn i) =
          rmsSpec s X QW RMS M N BM BN sxb sxm sxk sqwk sqwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem KOut (outOffset s N BM BN skb skm skn i) =
          rmsSpec s X KW RMS M N BM BN sxb sxm sxk skwk skwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem V (outOffset s N BM BN svb svm svn i) =
          rmsSpec s X VW RMS M N BM BN sxb sxm sxk svwk svwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ r o,
        (r ≠ Q ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i → o ≠ outOffset s N BM BN sqb sqm sqn i) →
        (r ≠ KOut ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i → o ≠ outOffset s N BM BN skb skm skn i) →
        (r ≠ V ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i → o ≠ outOffset s N BM BN svb svm svn i) →
        t.mem r o = s.mem r o) := by
  obtain ⟨s1, h1, hp1, hu1, hv1, hf1⟩ := RealExecution.run_body variant X QW RMS Q s
    M N K sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK EPS hu hiQ
  have hiK1 : ∀ i j : TileIndex [BM, BN], active s1 M N BM BN i → active s1 M N BM BN j →
      outOffset s1 N BM BN skb skm skn i = outOffset s1 N BM BN skb skm skn j → i = j := by
    simpa only [active_congr hp1, outOffset_congr hp1] using hiK
  obtain ⟨s2, h2, hp2, hu2, hv2, hf2⟩ := RealExecution.run_body variant X KW RMS KOut s1
    M N K sxb sxm sxk skwk skwn srms skb skm skn BM BN BK EPS hu1 hiK1
  have hp2' : s2.pids = s.pids := hp2.trans hp1
  have hiV2 : ∀ i j : TileIndex [BM, BN], active s2 M N BM BN i → active s2 M N BM BN j →
      outOffset s2 N BM BN svb svm svn i = outOffset s2 N BM BN svb svm svn j → i = j := by
    simpa only [active_congr hp2', outOffset_congr hp2'] using hiV
  obtain ⟨s3, h3, hp3, hu3, hv3, hf3⟩ := RealExecution.run_body variant X VW RMS V s2
    M N K sxb sxm sxk svwk svwn srms svb svm svn BM BN BK EPS hu2 hiV2
  refine ⟨s3, ?_, hp3.trans hp2', hu3, ?_, ?_, ?_, ?_⟩
  · rw [exec, ComputeKernel.toAlgKernel, projection]
    change stepStmts (body Bool.false variant X QW KW VW RMS Q KOut V M N K
      sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK EPS) s = some s3
    rw [body, List.append_assoc, stepStmts.append_some h1, stepStmts.append_some h2, h3]
  · intro i hi
    have hQ : ∀ o, s3.mem Q o = s1.mem Q o := fun o =>
      (hf3 Q o (Or.inl hQV)).trans (hf2 Q o (Or.inl hQK))
    simpa only [BlockState.readMem, hQ] using hv1 i hi
  · intro i hi
    have hK : ∀ o, s3.mem KOut o = s2.mem KOut o := fun o => hf3 KOut o (Or.inl hKV)
    have hval := hv2 i ((active_congr hp1 i).mpr hi)
    rw [outOffset_congr hp1 i] at hval
    rw [rmsSpec_congr hp1 (fun o => hf1 X o (Or.inl hXQ))
      (fun o => hf1 KW o (Or.inl hKWQ)) (fun o => hf1 RMS o (Or.inl hRQ))] at hval
    simpa only [BlockState.readMem, hK] using hval
  · intro i hi
    have hval := hv3 i ((active_congr hp2' i).mpr hi)
    rw [outOffset_congr hp2' i] at hval
    rw [rmsSpec_congr hp2'
      (fun o => (hf2 X o (Or.inl hXK)).trans (hf1 X o (Or.inl hXQ)))
      (fun o => (hf2 VW o (Or.inl hVWK)).trans (hf1 VW o (Or.inl hVWQ)))
      (fun o => (hf2 RMS o (Or.inl hRK)).trans (hf1 RMS o (Or.inl hRQ)))] at hval
    exact hval
  · intro r o hq hk hv
    rw [hf3 r o (by simpa only [active_congr hp2', outOffset_congr hp2'] using hv),
      hf2 r o (by simpa only [active_congr hp1, outOffset_congr hp1] using hk), hf1 r o hq]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.QKVExecution
