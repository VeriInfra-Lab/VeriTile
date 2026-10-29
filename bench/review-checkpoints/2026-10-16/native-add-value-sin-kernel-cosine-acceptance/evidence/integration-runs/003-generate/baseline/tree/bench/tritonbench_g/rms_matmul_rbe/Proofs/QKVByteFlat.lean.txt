import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKVByte
import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKVFlat
import bench.tritonbench_g.rms_matmul_rbe.Proofs.FP8Flat

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.QKVByteFlat
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

theorem body_bytes (variant : Bool) (bounds : RegionBounds)
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat) (EPS : ℝ)
    (hu : ∀ r o, s.undef r o = 0)
    (hXW : X ≠ W) (hXR : X ≠ RMS)
    (hbytes : ByteExecution.Bytes s W RMS N BN BK swk swn srms (cdiv K BK))
    (hb : AccessSafety.Reads bounds s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < bounds OUT) :
    Stmt.TraceSafeList bounds (RealSource.body Bool.true variant X W RMS OUT M N K
      sxb sxm sxk swk swn srms sob som son BM BN BK EPS) s := by
  have h := ByteExecution.complete .fp16 Bool.true variant bounds X W RMS OUT s
    M N K sxb sxm sxk swk swn srms sob som son BM BN BK 0 Bool.true 0 EPS hu hXW hXR hbytes hb ho
  rw [ComputeKernel.toAlgKernel, RealSource.projection] at h
  exact h

/-- All three complete passes are safe under explicit padded-load and active
output bounds. Earlier passes preserve the program ids used by each later one. -/
theorem safe_bytes (input : TypedSource.Storage) (variant : Bool) (bounds : RegionBounds)
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
    (hQK : Q ≠ KOut) (hQV : Q ≠ V) (hKV : KOut ≠ V)
    (bq : AccessSafety.Reads bounds s X QW RMS M N K BM BN BK sxb sxm sxk sqwk sqwn srms)
    (bk : AccessSafety.Reads bounds s X KW RMS M N K BM BN BK sxb sxm sxk skwk skwn srms)
    (bv : AccessSafety.Reads bounds s X VW RMS M N K BM BN BK sxb sxm sxk svwk svwn srms)
    (oq : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN sqb sqm sqn i < bounds Q)
    (ok : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN skb skm skn i < bounds KOut)
    (ov : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN svb svm svn i < bounds V) :
    (TypedSource.qkv input Bool.true variant X QW KW VW RMS Q KOut V M N K
      sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨s1, h1, hp1, hu1, hv1, hf1⟩ := ByteExecution.run_body variant X QW RMS Q s
    M N K sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK EPS hu hXQW hXR bytesQ hiQ
  have hiK1 : ∀ i j : TileIndex [BM, BN], active s1 M N BM BN i → active s1 M N BM BN j →
      outOffset s1 N BM BN skb skm skn i = outOffset s1 N BM BN skb skm skn j → i = j := by
    simpa only [active_congr hp1, outOffset_congr hp1] using hiK
  have bytesK1 := (QKVByte.bytes_congr hp1 (fun o => hf1 KW o (Or.inl hKWQ))
    (fun o => hf1 RMS o (Or.inl hRQ))).mpr bytesK
  obtain ⟨s2, h2, hp2, hu2, hv2, hf2⟩ := ByteExecution.run_body variant X KW RMS KOut s1
    M N K sxb sxm sxk skwk skwn srms skb skm skn BM BN BK EPS hu1 hXKW hXR bytesK1 hiK1
  have hp2' : s2.pids = s.pids := hp2.trans hp1
  have hiV2 : ∀ i j : TileIndex [BM, BN], active s2 M N BM BN i → active s2 M N BM BN j →
      outOffset s2 N BM BN svb svm svn i = outOffset s2 N BM BN svb svm svn j → i = j := by
    simpa only [active_congr hp2', outOffset_congr hp2'] using hiV
  have bytesV2 := (QKVByte.bytes_congr hp2'
    (fun o => (hf2 VW o (Or.inl hVWK)).trans (hf1 VW o (Or.inl hVWQ)))
    (fun o => (hf2 RMS o (Or.inl hRK)).trans (hf1 RMS o (Or.inl hRQ)))).mpr bytesV
  rw [ComputeKernel.toAlgKernel, QKVExecution.projection]
  change Stmt.TraceSafeList bounds (QKVExecution.body Bool.true variant X QW KW VW RMS Q KOut V M N K
    sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
    sqb sqm sqn skb skm skn svb svm svn BM BN BK EPS) s
  rw [QKVExecution.body, List.append_assoc]
  apply TraceProofs.append_safe _ s
    (body_bytes variant bounds X QW RMS Q s M N K sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK EPS hu hXQW hXR bytesQ bq oq)
  intro t ht
  have he : t = s1 := Option.some.inj (ht.symm.trans h1)
  subst t
  apply TraceProofs.append_safe _ s1
    (body_bytes variant bounds X KW RMS KOut s1 M N K sxb sxm sxk skwk skwn srms skb skm skn BM BN BK EPS hu1 hXKW hXR bytesK1
      (by simpa only [AccessSafety.Reads, rowIndex, colIndex, rowGlobal, colGlobal, hp1] using bk)
      (by simpa only [active_congr hp1, outOffset_congr hp1] using ok))
  intro t ht
  have he : t = s2 := Option.some.inj (ht.symm.trans h2)
  subst t
  exact body_bytes variant bounds X VW RMS V s2 M N K sxb sxm sxk svwk svwn srms svb svm svn BM BN BK EPS hu2 hXVW hXR bytesV2
    (by simpa only [AccessSafety.Reads, rowIndex, colIndex, rowGlobal, colGlobal, hp2'] using bv)
    (by simpa only [active_congr hp2', outOffset_congr hp2'] using ov)

theorem bytes (input : TypedSource.Storage) (variant : Bool) (A : FlatAlloc)
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
    (hXQW : X ≠ QW) (hXKW : X ≠ KW) (hXVW : X ≠ VW) (hXR : X ≠ RMS)
    (bytesQ : ByteExecution.Bytes s QW RMS N BN BK sqwk sqwn srms (cdiv K BK))
    (bytesK : ByteExecution.Bytes s KW RMS N BN BK skwk skwn srms (cdiv K BK))
    (bytesV : ByteExecution.Bytes s VW RMS N BN BK svwk svwn srms (cdiv K BK))
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
    ∃ t, exec (A.flattenKernel (TypedSource.qkv input Bool.true variant X QW KW VW RMS Q KOut V M N K
        sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS).eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr Q (outOffset s N BM BN sqb sqm sqn i)) =
          rmsSpec (ByteLoads.view QW RMS s) X QW RMS M N BM BN sxb sxm sxk sqwk sqwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr KOut (outOffset s N BM BN skb skm skn i)) =
          rmsSpec (ByteLoads.view KW RMS s) X KW RMS M N BM BN sxb sxm sxk skwk skwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr V (outOffset s N BM BN svb svm svn i)) =
          rmsSpec (ByteLoads.view VW RMS s) X VW RMS M N BM BN sxb sxm sxk svwk svwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ r off, (r ≠ A.flat ∨ ∀ region offset,
        QKVFlat.Written s Q KOut V M N BM BN sqb sqm sqn skb skm skn svb svm svn region offset →
        off ≠ A.addr region offset) → t.mem r off = (A.flattenState s).mem r off) := by
  have hu' : ∀ r o, s.undef r o = 0 := by intro r o; rw [hu]
  obtain ⟨t, he, _hp, _hu, hq, hk, hv, hf⟩ := QKVByte.run input variant
    X QW KW VW RMS Q KOut V s M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
    sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS hu' hiQ hiK hiV
    hXQW hXKW hXVW hXR bytesQ bytesK bytesV
    hXQ hRQ hKWQ hVWQ hXK hRK hVWK hQK hQV hKV
  have hs := safe_bytes input variant A.extent X QW KW VW RMS Q KOut V s M N K
    sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn BM BN BK
    start theta EPS hu' hiQ hiK hiV hXQW hXKW hXVW hXR bytesQ bytesK bytesV
    hXQ hRQ hKWQ hVWQ hXK hRK hVWK hQK hQV hKV bq bk bv oq ok ov
  refine ⟨A.flattenState t, ?_, ?_, ?_, ?_, ?_⟩
  · rw [A.exec_flatten hd hc _ s hs
      (QKVFlat.flattenOk input Bool.true variant X QW KW VW RMS Q KOut V M N K
        sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS) hu,
      he, Option.map_some]
  · intro i hi
    have hr : Q ∈ A.regions := by
      by_contra hn
      have hz := hc Q hn
      have hb := oq i hi
      omega
    have hh := A.flattenState_readMemAs hd t hr (oq i hi) .real
    simp only [BlockState.readMemAs_real] at hh
    exact (Option.some.inj hh).trans (hq i hi)
  · intro i hi
    have hr : KOut ∈ A.regions := by
      by_contra hn
      have hz := hc KOut hn
      have hb := ok i hi
      omega
    have hh := A.flattenState_readMemAs hd t hr (ok i hi) .real
    simp only [BlockState.readMemAs_real] at hh
    exact (Option.some.inj hh).trans (hk i hi)
  · intro i hi
    have hr : V ∈ A.regions := by
      by_contra hn
      have hz := hc V hn
      have hb := ov i hi
      omega
    have hh := A.flattenState_readMemAs hd t hr (ov i hi) .real
    simp only [BlockState.readMemAs_real] at hh
    exact (Option.some.inj hh).trans (hv i hi)
  · apply QKVFlat.frame
    intro r o hn
    apply hf r o
    · by_cases hr : r = Q
      · right
        intro i hi he
        exact hn ⟨i, hi, Or.inl ⟨hr, he⟩⟩
      · exact Or.inl hr
    · by_cases hr : r = KOut
      · right
        intro i hi he
        exact hn ⟨i, hi, Or.inr (Or.inl ⟨hr, he⟩)⟩
      · exact Or.inl hr
    · by_cases hr : r = V
      · right
        intro i hi he
        exact hn ⟨i, hi, Or.inr (Or.inr ⟨hr, he⟩)⟩
      · exact Or.inl hr

theorem arrays (input : TypedSource.Storage) (variant : Bool) (A : FlatAlloc)
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
    ∃ t, exec (A.flattenKernel (TypedSource.qkv input Bool.true variant X QW KW VW RMS Q KOut V M N K
        sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS).eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr Q (outOffset s N BM BN sqb sqm sqn i)) =
          PaddedRMSGemm.output K (cdiv K BK) BM BN BK EPS xs
            (fun t e => (rs t e).value) (fun t e j => (qws t e j).value) i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr KOut (outOffset s N BM BN skb skm skn i)) =
          PaddedRMSGemm.output K (cdiv K BK) BM BN BK EPS xs
            (fun t e => (rs t e).value) (fun t e j => (kws t e j).value) i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr V (outOffset s N BM BN svb svm svn i)) =
          PaddedRMSGemm.output K (cdiv K BK) BM BN BK EPS xs
            (fun t e => (rs t e).value) (fun t e j => (vws t e j).value) i.1 i.2.1) ∧
      (∀ r off, (r ≠ A.flat ∨ ∀ region offset,
        QKVFlat.Written s Q KOut V M N BM BN sqb sqm sqn skb skm skn svb svm svn region offset →
        off ≠ A.addr region offset) → t.mem r off = (A.flattenState s).mem r off) := by
  have bytesQ := ByteExecution.inputs_bytes s X QW RMS M N K BM BN BK sxb sxm sxk sqwk sqwn srms xs rs qws inputQ
  have bytesK := ByteExecution.inputs_bytes s X KW RMS M N K BM BN BK sxb sxm sxk skwk skwn srms xs rs kws inputK
  have bytesV := ByteExecution.inputs_bytes s X VW RMS M N K BM BN BK sxb sxm sxk svwk svwn srms xs rs vws inputV
  obtain ⟨t, he, hq, hk, hv, hf⟩ := bytes input variant A X QW KW VW RMS Q KOut V s M N K
    sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn BM BN BK
    start theta EPS hu hiQ hiK hiV hXQW hXKW hXVW hXR bytesQ bytesK bytesV
    hXQ hRQ hKWQ hVWQ hXK hRK hVWK hQK hQV hKV hd hc bq bk bv oq ok ov
  refine ⟨t, he, ?_, ?_, ?_, hf⟩
  · intro i hi
    exact (hq i hi).trans (ByteExecution.formula s X QW RMS M N K BM BN BK
      sxb sxm sxk sqwk sqwn srms EPS xs rs qws hXQW hXR inputQ i.1 i.2.1)
  · intro i hi
    exact (hk i hi).trans (ByteExecution.formula s X KW RMS M N K BM BN BK
      sxb sxm sxk skwk skwn srms EPS xs rs kws hXKW hXR inputK i.1 i.2.1)
  · intro i hi
    exact (hv i hi).trans (ByteExecution.formula s X VW RMS M N K BM BN BK
      sxb sxm sxk svwk svwn srms EPS xs rs vws hXVW hXR inputV i.1 i.2.1)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.QKVByteFlat
