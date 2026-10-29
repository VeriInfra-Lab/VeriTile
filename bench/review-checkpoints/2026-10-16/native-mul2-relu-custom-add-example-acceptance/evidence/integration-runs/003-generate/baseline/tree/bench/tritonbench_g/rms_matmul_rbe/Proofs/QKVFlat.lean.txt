import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKV
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Flat
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Geometry

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.QKVFlat
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

theorem body_half (variant : Bool) (bounds : RegionBounds)
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat) (EPS : ℝ)
    (hu : ∀ r o, s.undef r o = 0)
    (hb : AccessSafety.Reads bounds s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < bounds OUT) :
    Stmt.TraceSafeList bounds (RealSource.body Bool.false variant X W RMS OUT M N K
      sxb sxm sxk swk swn srms sob som son BM BN BK EPS) s := by
  have h := AccessSafety.complete_half .fp16 Bool.false variant bounds X W RMS OUT s
    M N K sxb sxm sxk swk swn srms sob som son BM BN BK 0 Bool.false 0 EPS hu hb ho
  rw [ComputeKernel.toAlgKernel, RealSource.projection] at h
  exact h

/-- All three complete passes are safe under explicit padded-load and active
output bounds. Earlier passes preserve the program ids used by each later one. -/
theorem safe_half (input : TypedSource.Storage) (variant : Bool) (bounds : RegionBounds)
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
    (bq : AccessSafety.Reads bounds s X QW RMS M N K BM BN BK sxb sxm sxk sqwk sqwn srms)
    (bk : AccessSafety.Reads bounds s X KW RMS M N K BM BN BK sxb sxm sxk skwk skwn srms)
    (bv : AccessSafety.Reads bounds s X VW RMS M N K BM BN BK sxb sxm sxk svwk svwn srms)
    (oq : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN sqb sqm sqn i < bounds Q)
    (ok : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN skb skm skn i < bounds KOut)
    (ov : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN svb svm svn i < bounds V) :
    (TypedSource.qkv input Bool.false variant X QW KW VW RMS Q KOut V M N K
      sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨s1, h1, hp1, hu1, hv1, hf1⟩ := RealExecution.run_body variant X QW RMS Q s
    M N K sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK EPS hu hiQ
  have hiK1 : ∀ i j : TileIndex [BM, BN], active s1 M N BM BN i → active s1 M N BM BN j →
      outOffset s1 N BM BN skb skm skn i = outOffset s1 N BM BN skb skm skn j → i = j := by
    simpa only [active_congr hp1, outOffset_congr hp1] using hiK
  obtain ⟨s2, h2, hp2, hu2, hv2, hf2⟩ := RealExecution.run_body variant X KW RMS KOut s1
    M N K sxb sxm sxk skwk skwn srms skb skm skn BM BN BK EPS hu1 hiK1
  have hp2' : s2.pids = s.pids := hp2.trans hp1
  rw [ComputeKernel.toAlgKernel, QKVExecution.projection]
  change Stmt.TraceSafeList bounds (QKVExecution.body Bool.false variant X QW KW VW RMS Q KOut V M N K
    sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
    sqb sqm sqn skb skm skn svb svm svn BM BN BK EPS) s
  rw [QKVExecution.body, List.append_assoc]
  apply TraceProofs.append_safe _ s
    (body_half variant bounds X QW RMS Q s M N K sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK EPS hu bq oq)
  intro t ht
  have he : t = s1 := Option.some.inj (ht.symm.trans h1)
  subst t
  apply TraceProofs.append_safe _ s1
    (body_half variant bounds X KW RMS KOut s1 M N K sxb sxm sxk skwk skwn srms skb skm skn BM BN BK EPS hu1
      (by simpa only [AccessSafety.Reads, rowIndex, colIndex, rowGlobal, colGlobal, hp1] using bk)
      (by simpa only [active_congr hp1, outOffset_congr hp1] using ok))
  intro t ht
  have he : t = s2 := Option.some.inj (ht.symm.trans h2)
  subst t
  exact body_half variant bounds X VW RMS V s2 M N K sxb sxm sxk svwk svwn srms svb svm svn BM BN BK EPS hu2
    (by simpa only [AccessSafety.Reads, rowIndex, colIndex, rowGlobal, colGlobal, hp2'] using bv)
    (by simpa only [active_congr hp2', outOffset_congr hp2'] using ov)

theorem flattenOk (input : TypedSource.Storage) (fp8 variant : Bool)
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ) :
    (TypedSource.qkv input fp8 variant X QW KW VW RMS Q KOut V M N K
      sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS).eraseDType.toAlgKernel.FlattenOk := by
  rw [ComputeKernel.toAlgKernel, QKVExecution.projection]
  cases fp8 <;> cases variant <;>
    simp [Kernel.FlattenOk, QKVExecution.body, RealSource.body, RealSource.loop, RealSource.tail,
      RealSource.weightLoad, RealSource.ceilDivOp, rrmPrologue, rrmLoopBody, rrmStoreTail,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def,
      Stmt.eraseDType, Op.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
      FloatDType.toTileDType, TileShape.eraseAxis]

/-- Generic raw frame transfer also frames unused cells within allocations,
unallocated addresses and cells whose runtime storage type is unrelated. -/
theorem frame (A : FlatAlloc) (s t : BlockState) (written : RegionName → Nat → Prop)
    (hf : ∀ r o, ¬ written r o → t.mem r o = s.mem r o)
    (r : RegionName) (off : Nat)
    (hn : r ≠ A.flat ∨ ∀ region offset, written region offset → off ≠ A.addr region offset) :
    (A.flattenState t).mem r off = (A.flattenState s).mem r off := by
  change (if r = A.flat then A.readFlat t off else MemCell.real 0) =
    (if r = A.flat then A.readFlat s off else MemCell.real 0)
  by_cases hr : r = A.flat
  · subst r
    simp only [if_pos rfl]
    unfold FlatAlloc.readFlat
    cases hc : A.decode off with
    | none => rfl
    | some p =>
      apply congrArg A.trCell
      apply hf p.1 p.2
      intro hw
      rcases hn with hn | hn
      · exact hn rfl
      · exact hn p.1 p.2 hw (A.decode_sound hc).2.1
  · simp [hr]

/-- The union of all three active source output footprints. -/
def Written (s : BlockState) (Q KOut V : RegionName)
    (M N BM BN sqb sqm sqn skb skm skn svb svm svn : Nat) (r : RegionName) (o : Nat) : Prop :=
  ∃ i : TileIndex [BM, BN], active s M N BM BN i ∧
    ((r = Q ∧ o = outOffset s N BM BN sqb sqm sqn i) ∨
     (r = KOut ∧ o = outOffset s N BM BN skb skm skn i) ∨
     (r = V ∧ o = outOffset s N BM BN svb svm svn i))

theorem half (input : TypedSource.Storage) (variant : Bool) (A : FlatAlloc)
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
    ∃ t, exec (A.flattenKernel (TypedSource.qkv input Bool.false variant X QW KW VW RMS Q KOut V M N K
        sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS).eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr Q (outOffset s N BM BN sqb sqm sqn i)) =
          rmsSpec s X QW RMS M N BM BN sxb sxm sxk sqwk sqwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr KOut (outOffset s N BM BN skb skm skn i)) =
          rmsSpec s X KW RMS M N BM BN sxb sxm sxk skwk skwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ i : TileIndex [BM, BN], active s M N BM BN i →
        t.readMem A.flat (A.addr V (outOffset s N BM BN svb svm svn i)) =
          rmsSpec s X VW RMS M N BM BN sxb sxm sxk svwk svwn srms BK (cdiv K BK) K EPS i.1 i.2.1) ∧
      (∀ r off, (r ≠ A.flat ∨ ∀ region offset,
        Written s Q KOut V M N BM BN sqb sqm sqn skb skm skn svb svm svn region offset →
        off ≠ A.addr region offset) → t.mem r off = (A.flattenState s).mem r off) := by
  have hu' : ∀ r o, s.undef r o = 0 := by intro r o; rw [hu]
  obtain ⟨t, he, _hp, _hu, hq, hk, hv, hf⟩ := QKVExecution.run input variant
    X QW KW VW RMS Q KOut V s M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
    sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS hu' hiQ hiK hiV
    hXQ hRQ hKWQ hVWQ hXK hRK hVWK hQK hQV hKV
  have hs := safe_half input variant A.extent X QW KW VW RMS Q KOut V s M N K
    sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn BM BN BK
    start theta EPS hu' hiQ hiK hiV bq bk bv oq ok ov
  refine ⟨A.flattenState t, ?_, ?_, ?_, ?_, ?_⟩
  · rw [A.exec_flatten hd hc _ s hs
      (flattenOk input Bool.false variant X QW KW VW RMS Q KOut V M N K
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
  · apply frame
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

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.QKVFlat
