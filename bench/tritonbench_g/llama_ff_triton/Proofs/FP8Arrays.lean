import bench.tritonbench_g.llama_ff_triton.Proofs.FP8Execution

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.ByteExecution
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

/-- Independent input arrays for the byte-storage source. Each weight premise
requires an actual integer cell with the supplied finite signed-byte payload;
no untyped/default memory read can discharge it. All padded K reads are included. -/
def Inputs (s : BlockState) (A W1 W3 RMS : RegionName)
    (M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms : Nat)
    (xs : Fin (Mathematics.steps K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (Mathematics.steps K BK) → Fin BK → Mathematics.Byte)
    (w1 w3 : Fin (Mathematics.steps K BK) → Fin BK → Fin BN → Mathematics.Byte) : Prop :=
  (∀ t i e, aElem s A M N BM BN sam sak i (t.val * BK + e.val) = xs t i e) ∧
  (∀ t e, s.readMemTyped .int RMS ((t.val * BK + e.val) * srms) = some (rs t e).val) ∧
  (∀ t e j, s.readMemTyped .int W1 ((t.val * BK + e.val) * sw1k + colIndex s N BN j * sw1n) = some (w1 t e j).val) ∧
  (∀ t e j, s.readMemTyped .int W3 ((t.val * BK + e.val) * sw3k + colIndex s N BN j * sw3n) = some (w3 t e j).val)

theorem finite_of_typed (s : BlockState) (r : RegionName) (o : Nat) (b : Mathematics.Byte)
    (h : s.readMemTyped .int r o = some b.val) : ByteLoads.FiniteAt s r o := by
  simpa [ByteLoads.FiniteAt, BlockState.readMemValue, h] using b.property

theorem view_of_typed (W1 W3 RMS : RegionName) (s : BlockState) (r : RegionName)
    (o : Nat) (b : Mathematics.Byte) (hw : r = W1 ∨ r = W3 ∨ r = RMS)
    (h : s.readMemTyped .int r o = some b.val) :
    (ByteLoads.view W1 W3 RMS s).readMem r o = b.value := by
  have hv := ByteLoads.decoded_view W1 W3 RMS s r o hw (finite_of_typed s r o b h)
  simp only [BlockState.readMemValue, h, b.decoded] at hv
  exact (Option.some.inj hv).symm

theorem inputs_bytes (s : BlockState) (A W1 W3 RMS : RegionName)
    (M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms : Nat)
    (xs : Fin (Mathematics.steps K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (Mathematics.steps K BK) → Fin BK → Mathematics.Byte)
    (w1 w3 : Fin (Mathematics.steps K BK) → Fin BK → Fin BN → Mathematics.Byte)
    (h : Inputs s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms xs rs w1 w3) :
    Bytes s W1 W3 RMS N BN BK sw1k sw1n sw3k sw3n srms (cdiv K BK) := by
  refine ⟨?_, ?_, ?_⟩
  · intro t ht e j
    have hh := finite_of_typed _ _ _ _ (h.2.2.1 ⟨t, ht⟩ e j)
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hh
  · intro t ht e j
    have hh := finite_of_typed _ _ _ _ (h.2.2.2 ⟨t, ht⟩ e j)
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hh
  · intro t ht e
    have hh := finite_of_typed _ _ _ _ (h.2.1 ⟨t, ht⟩ e)
    simpa [Nat.add_mul, Nat.add_comm] using hh

theorem inputs_view (s : BlockState) (A W1 W3 RMS : RegionName)
    (M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms : Nat)
    (xs : Fin (Mathematics.steps K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (Mathematics.steps K BK) → Fin BK → Mathematics.Byte)
    (w1 w3 : Fin (Mathematics.steps K BK) → Fin BK → Fin BN → Mathematics.Byte)
    (hA1 : A ≠ W1) (hA3 : A ≠ W3) (hAR : A ≠ RMS)
    (h : Inputs s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms xs rs w1 w3) :
    RealExecution.Inputs (ByteLoads.view W1 W3 RMS s) A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms
      xs (fun t e => (rs t e).value) (fun t e j => (w1 t e j).value) (fun t e j => (w3 t e j).value) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro t i e
    have he : aElem (ByteLoads.view W1 W3 RMS s) A M N BM BN sam sak i (t.val * BK + e.val) =
        aElem s A M N BM BN sam sak i (t.val * BK + e.val) := by
      simp only [aElem, rowIndex, rowGlobal, ByteLoads.view, BlockState.readMem,
        hA1, hA3, hAR, false_or, ↓reduceIte]
    exact he.trans (h.1 t i e)
  · intro t e
    exact view_of_typed W1 W3 RMS s RMS _ _ (by simp) (h.2.1 t e)
  · intro t e j
    exact view_of_typed W1 W3 RMS s W1 _ _ (by simp) (h.2.2.1 t e j)
  · intro t e j
    exact view_of_typed W1 W3 RMS s W3 _ _ (by simp) (h.2.2.2 t e j)

/-- Both complete byte-source variants realize the independent decoded-array
formula, retain actual integer storage, and preserve every nonwritten typed cell. -/
theorem arrays (variant : Bool) (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ)
    (xs : Fin (Mathematics.steps K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (Mathematics.steps K BK) → Fin BK → Mathematics.Byte)
    (w1 w3 : Fin (Mathematics.steps K BK) → Fin BK → Fin BN → Mathematics.Byte)
    (hu : ∀ rg o, s.undef rg o = 0) (hc : 0 < soutn) (hr : N * soutn ≤ soutm)
    (hA1 : A ≠ W1) (hA3 : A ≠ W3) (hAR : A ≠ RMS)
    (hinput : Inputs s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms xs rs w1 w3) :
    ∃ t, exec (TypedSource.kernel Bool.true variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).eraseDType s = some t ∧
      (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        t.readMem OUT (outOffset s N BM BN soutm soutn idx) =
          Mathematics.output K BK BM BN EPS xs (fun t e => (rs t e).value)
            (fun t e j => (w1 t e j).value) (fun t e j => (w3 t e j).value) idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        o ≠ outOffset s N BM BN soutm soutn idx) → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := run variant A W1 W3 OUT RMS s M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS hu hA1 hA3 hAR
    (inputs_bytes s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms xs rs w1 w3 hinput)
    (Geometry.active_injective s M N BM BN soutm soutn hc hr)
  refine ⟨t, he, ?_, hf⟩
  intro idx hi
  rw [hv idx hi, RealExecution.formula _ A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms EPS xs _ _ _
    (inputs_view s A W1 W3 RMS M N K BM BN BK sam sak sw1k sw1n sw3k sw3n srms xs rs w1 w3 hA1 hA3 hAR hinput)]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.ByteExecution
