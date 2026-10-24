import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKV
import bench.tritonbench_g.rms_matmul_rbe.Proofs.FP8Safety

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.QKVByte
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Finite-decode obligations depend only on program ids and the raw input
cells used by that pass, so earlier disjoint stores preserve them. -/
theorem bytes_congr {s t : BlockState} {W RMS : RegionName}
    {N BN BK swk swn srms blocks : Nat} (hp : s.pids = t.pids)
    (hw : ∀ o, s.mem W o = t.mem W o) (hr : ∀ o, s.mem RMS o = t.mem RMS o) :
    ByteExecution.Bytes s W RMS N BN BK swk swn srms blocks ↔
      ByteExecution.Bytes t W RMS N BN BK swk swn srms blocks := by
  simp only [ByteExecution.Bytes, ByteLoads.FiniteAt, colIndex, colGlobal, hp,
    BlockState.readMemValue, BlockState.readMemTyped, hw, hr]

/-- A representation view is local to each cell. It never uses values written
by a preceding pass when those raw input cells were framed. -/
theorem view_cell {s t : BlockState} (W RMS r : RegionName) (o : Nat)
    (h : s.mem r o = t.mem r o) :
    (ByteLoads.view W RMS s).mem r o = (ByteLoads.view W RMS t).mem r o := by
  simp only [ByteLoads.view, BlockState.readMemValue, BlockState.readMemTyped, h]

theorem formula_congr {s t : BlockState} {X W RMS : RegionName}
    {M N BM BN sxb sxm sxk swk swn srms BK blocks K : Nat} {EPS : ℝ}
    (hp : s.pids = t.pids) (hx : ∀ o, s.mem X o = t.mem X o)
    (hw : ∀ o, s.mem W o = t.mem W o) (hr : ∀ o, s.mem RMS o = t.mem RMS o)
    (i : Fin BM) (j : Fin BN) :
    rmsSpec (ByteLoads.view W RMS s) X W RMS M N BM BN sxb sxm sxk swk swn srms BK blocks K EPS i j =
      rmsSpec (ByteLoads.view W RMS t) X W RMS M N BM BN sxb sxm sxk swk swn srms BK blocks K EPS i j := by
  exact rmsSpec_congr (sA := ByteLoads.view W RMS s) (sB := ByteLoads.view W RMS t) hp
    (fun o => view_cell W RMS X o (hx o))
    (fun o => view_cell W RMS W o (hw o)) (fun o => view_cell W RMS RMS o (hr o))

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
    (hXQW : X ≠ QW) (hXKW : X ≠ KW) (hXVW : X ≠ VW) (hXR : X ≠ RMS)
    (bytesQ : ByteExecution.Bytes s QW RMS N BN BK sqwk sqwn srms (cdiv K BK))
    (bytesK : ByteExecution.Bytes s KW RMS N BN BK skwk skwn srms (cdiv K BK))
    (bytesV : ByteExecution.Bytes s VW RMS N BN BK svwk svwn srms (cdiv K BK))
    (hXQ : X ≠ Q) (hRQ : RMS ≠ Q) (hKWQ : KW ≠ Q) (hVWQ : VW ≠ Q)
    (hXK : X ≠ KOut) (hRK : RMS ≠ KOut) (hVWK : VW ≠ KOut)
    (hQK : Q ≠ KOut) (hQV : Q ≠ V) (hKV : KOut ≠ V) :
    ∃ t, exec (TypedSource.qkv input Bool.true variant X QW KW VW RMS Q KOut V M N K
        sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS).eraseDType s = some t ∧
      t.pids = s.pids ∧ (∀ r o, t.undef r o = 0) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem Q (outOffset s N BM BN sqb sqm sqn i) =
          rmsSpec (ByteLoads.view QW RMS s) X QW RMS M N BM BN sxb sxm sxk sqwk sqwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem KOut (outOffset s N BM BN skb skm skn i) =
          rmsSpec (ByteLoads.view KW RMS s) X KW RMS M N BM BN sxb sxm sxk skwk skwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem V (outOffset s N BM BN svb svm svn i) =
          rmsSpec (ByteLoads.view VW RMS s) X VW RMS M N BM BN sxb sxm sxk svwk svwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ r o,
        (r ≠ Q ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i → o ≠ outOffset s N BM BN sqb sqm sqn i) →
        (r ≠ KOut ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i → o ≠ outOffset s N BM BN skb skm skn i) →
        (r ≠ V ∨ ∀ i : TileIndex [BM, BN], active s M N BM BN i → o ≠ outOffset s N BM BN svb svm svn i) →
        t.mem r o = s.mem r o) := by
  obtain ⟨s1, h1, hp1, hu1, hv1, hf1⟩ := ByteExecution.run_body variant X QW RMS Q s
    M N K sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK EPS hu hXQW hXR bytesQ hiQ
  have hiK1 : ∀ i j : TileIndex [BM, BN], active s1 M N BM BN i → active s1 M N BM BN j →
      outOffset s1 N BM BN skb skm skn i = outOffset s1 N BM BN skb skm skn j → i = j := by
    simpa only [active_congr hp1, outOffset_congr hp1] using hiK
  have bytesK1 := (bytes_congr hp1 (fun o => hf1 KW o (Or.inl hKWQ))
    (fun o => hf1 RMS o (Or.inl hRQ))).mpr bytesK
  obtain ⟨s2, h2, hp2, hu2, hv2, hf2⟩ := ByteExecution.run_body variant X KW RMS KOut s1
    M N K sxb sxm sxk skwk skwn srms skb skm skn BM BN BK EPS hu1 hXKW hXR bytesK1 hiK1
  have hp2' : s2.pids = s.pids := hp2.trans hp1
  have hiV2 : ∀ i j : TileIndex [BM, BN], active s2 M N BM BN i → active s2 M N BM BN j →
      outOffset s2 N BM BN svb svm svn i = outOffset s2 N BM BN svb svm svn j → i = j := by
    simpa only [active_congr hp2', outOffset_congr hp2'] using hiV
  have bytesV2 := (bytes_congr hp2'
    (fun o => (hf2 VW o (Or.inl hVWK)).trans (hf1 VW o (Or.inl hVWQ)))
    (fun o => (hf2 RMS o (Or.inl hRK)).trans (hf1 RMS o (Or.inl hRQ)))).mpr bytesV
  obtain ⟨s3, h3, hp3, hu3, hv3, hf3⟩ := ByteExecution.run_body variant X VW RMS V s2
    M N K sxb sxm sxk svwk svwn srms svb svm svn BM BN BK EPS hu2 hXVW hXR bytesV2 hiV2
  refine ⟨s3, ?_, hp3.trans hp2', hu3, ?_, ?_, ?_, ?_⟩
  · rw [exec, ComputeKernel.toAlgKernel, QKVExecution.projection]
    change stepStmts (QKVExecution.body Bool.true variant X QW KW VW RMS Q KOut V M N K
      sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK EPS) s = some s3
    rw [QKVExecution.body, List.append_assoc, stepStmts.append_some h1, stepStmts.append_some h2, h3]
  · intro i hi
    have hQ : ∀ o, s3.mem Q o = s1.mem Q o := fun o =>
      (hf3 Q o (Or.inl hQV)).trans (hf2 Q o (Or.inl hQK))
    simpa only [BlockState.readMem, hQ] using hv1 i hi
  · intro i hi
    have hK : ∀ o, s3.mem KOut o = s2.mem KOut o := fun o => hf3 KOut o (Or.inl hKV)
    have hval := hv2 i ((active_congr hp1 i).mpr hi)
    rw [outOffset_congr hp1 i] at hval
    rw [formula_congr hp1 (fun o => hf1 X o (Or.inl hXQ))
      (fun o => hf1 KW o (Or.inl hKWQ)) (fun o => hf1 RMS o (Or.inl hRQ))] at hval
    simpa only [BlockState.readMem, hK] using hval
  · intro i hi
    have hval := hv3 i ((active_congr hp2' i).mpr hi)
    rw [outOffset_congr hp2' i] at hval
    rw [formula_congr hp2'
      (fun o => (hf2 X o (Or.inl hXK)).trans (hf1 X o (Or.inl hXQ)))
      (fun o => (hf2 VW o (Or.inl hVWK)).trans (hf1 VW o (Or.inl hVWQ)))
      (fun o => (hf2 RMS o (Or.inl hRK)).trans (hf1 RMS o (Or.inl hRQ)))] at hval
    exact hval
  · intro r o hq hk hv
    rw [hf3 r o (by simpa only [active_congr hp2', outOffset_congr hp2'] using hv),
      hf2 r o (by simpa only [active_congr hp1, outOffset_congr hp1] using hk), hf1 r o hq]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.QKVByte
