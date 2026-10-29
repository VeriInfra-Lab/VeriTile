import bench.tritonbench_g.llama_ff_triton.Proofs.FP8Arrays
import bench.tritonbench_g.llama_ff_triton.Proofs.FP8Safety

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.FlatIO
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem flattenOk (fp8 variant : Bool) (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ) :
    (TypedSource.kernel fp8 variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).eraseDType.toAlgKernel.FlattenOk := by
  rw [ComputeKernel.toAlgKernel, RealSource.projection]
  cases fp8 <;> cases variant <;>
    simp [Kernel.FlattenOk, RealSource.body, RealSource.loop, RealSource.tail,
      RealSource.weightLoad, RealSource.ceilDivOp, ffPrologue, ffLoopBody, ffStoreTail,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def,
      Stmt.eraseDType, Op.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
      FloatDType.toTileDType, TileShape.eraseAxis]

/-- Transport any proved regional real output and full cell frame through a
safe disjoint allocation. The input cell types are unrestricted. -/
theorem transport {α : Type} (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (k : Kernel) (hok : k.FlattenOk)
    (s : BlockState) (hu : s.undef = (fun _ _ => 0)) (hsafe : k.TraceSafe A.extent s)
    (Out : RegionName) (hout : Out ∈ A.regions) (write : α → Nat) (active : α → Prop)
    (expected : α → ℝ) (hb : ∀ i, active i → write i < A.extent Out)
    (hrun : ∃ t, exec k s = some t ∧ (∀ i, active i → t.readMem Out (write i) = expected i) ∧
      (∀ r o, (r ≠ Out ∨ ∀ i, active i → o ≠ write i) → t.mem r o = s.mem r o)) :
    ∃ t, exec (A.flattenKernel k) (A.flattenState s) = some t ∧
      (∀ i, active i → t.readMem A.flat (A.addr Out (write i)) = expected i) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ i, active i → o ≠ A.addr Out (write i)) →
        t.mem r o = (A.flattenState s).mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := hrun
  refine ⟨A.flattenState t, ?_, ?_, ?_⟩
  · rw [A.exec_flatten hd hcov k s hsafe hok hu, he, Option.map_some]
  · intro i hi
    have h := A.flattenState_readMemAs hd t hout (hb i hi) .real
    change (A.flattenState t).readMemAs .real A.flat (A.addr Out (write i)) =
      t.readMemAs .real Out (write i) at h
    rw [BlockState.readMemAs_real, BlockState.readMemAs_real] at h
    exact (Option.some.inj h).trans (hv i hi)
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
        obtain ⟨r, o⟩ := p
        obtain ⟨_, hoeq, _⟩ := A.decode_sound hdec
        show A.trCell (t.mem r o) = A.trCell (s.mem r o)
        refine congrArg A.trCell (hf r o ?_)
        by_cases hro : r = Out
        · subst hro
          refine Or.inr fun i hi => ?_
          rcases hcond with hflat | hn
          · exact absurd rfl hflat
          · intro heq
            exact hn i hi (by rw [hoeq, heq])
        · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

/-- Complete fp16-weight source execution over finite flat storage, with the
independent padded-array formula and the full frame of typed memory cells. -/
theorem half (variant : Bool) (alloc : FlatAlloc) (A W1 W3 OUT RMS : RegionName) (s : BlockState)
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
    ∃ t, exec (alloc.flattenKernel (TypedSource.kernel Bool.false variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).eraseDType.toAlgKernel)
        (alloc.flattenState s) = some t ∧
      (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        t.readMem alloc.flat (alloc.addr OUT (outOffset s N BM BN soutm soutn idx)) =
          Mathematics.output K BK BM BN EPS xs rs w1 w3 idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ alloc.flat ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        o ≠ alloc.addr OUT (outOffset s N BM BN soutm soutn idx)) →
        t.mem r o = (alloc.flattenState s).mem r o) := by
  apply transport alloc hd hcov _
    (flattenOk Bool.false variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS)
    s (funext fun r => funext fun o => hu r o) _ OUT hout
    (outOffset s N BM BN soutm soutn) (active s M N BM BN) _ ho
  · exact RealExecution.arrays variant A W1 W3 OUT RMS s M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS xs rs w1 w3 hu hc hr hinput
  · exact AccessSafety.complete_fp16 variant alloc.extent A W1 W3 OUT RMS s M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS hu hb ho

/-- Complete signed-byte E5M2 source execution over finite flat storage, with the
independent padded-array formula and the full frame of typed memory cells. -/
theorem bytes (variant : Bool) (alloc : FlatAlloc) (A W1 W3 OUT RMS : RegionName) (s : BlockState)
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
    ∃ t, exec (alloc.flattenKernel (TypedSource.kernel Bool.true variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).eraseDType.toAlgKernel)
        (alloc.flattenState s) = some t ∧
      (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        t.readMem alloc.flat (alloc.addr OUT (outOffset s N BM BN soutm soutn idx)) =
          Mathematics.output K BK BM BN EPS xs (fun t e => (rs t e).value) (fun t e j => (w1 t e j).value) (fun t e j => (w3 t e j).value) idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ alloc.flat ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        o ≠ alloc.addr OUT (outOffset s N BM BN soutm soutn idx)) →
        t.mem r o = (alloc.flattenState s).mem r o) := by
  apply transport alloc hd hcov _
    (flattenOk Bool.true variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS)
    s (funext fun r => funext fun o => hu r o) _ OUT hout
    (outOffset s N BM BN soutm soutn) (active s M N BM BN) _ ho
  · exact ByteExecution.arrays variant A W1 W3 OUT RMS s M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS xs rs w1 w3 hu hc hr hA1 hA3 hAR hinput
  · exact ByteExecution.complete variant alloc.extent A W1 W3 OUT RMS s M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS hu hA1 hA3 hAR
      (ByteExecution.inputs_bytes s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms xs rs w1 w3 hinput) hb ho

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.FlatIO
