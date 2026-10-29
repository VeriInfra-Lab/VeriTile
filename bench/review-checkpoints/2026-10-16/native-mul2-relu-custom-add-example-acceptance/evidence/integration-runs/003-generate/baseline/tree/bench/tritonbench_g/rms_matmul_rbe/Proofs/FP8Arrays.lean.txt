import bench.tritonbench_g.rms_matmul_rbe.Proofs.FP8Execution
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Geometry
import VeriTile.Triton.Math.PaddedRMSGemm

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.ByteExecution
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

/-- A signed byte with an actual finite E5M2 decoder result. -/
def Byte := { n : Int // Float8E5Bits.isInt8 n = Bool.true ∧
  (Float8E5Bits.ofInt n).decodeRat.isSome = Bool.true }

noncomputable def Byte.value (b : Byte) : ℝ :=
  (((Float8E5Bits.ofInt b.val).decodeRat.get b.property.2 : ℚ) : ℝ)

theorem Byte.decoded (b : Byte) :
    (Float8E5Bits.ofInt b.val).decodeRat.map (fun q : ℚ => (q : ℝ)) = some b.value := by
  cases h : (Float8E5Bits.ofInt b.val).decodeRat with
  | none => have := b.property.2; simp [h] at this
  | some q => simp [Byte.value, h]

/-- Independent arrays describe every actual unmasked read, including padded
K lanes. Byte premises require actual integer cells with the supplied payload;
an untyped/default read cannot establish this input relation. -/
def Inputs (s : BlockState) (X W RMS : RegionName)
    (M N K BM BN BK sxb sxm sxk swk swn srms : Nat)
    (xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (cdiv K BK) → Fin BK → Byte)
    (ws : Fin (cdiv K BK) → Fin BK → Fin BN → Byte) : Prop :=
  (∀ t i e, xElem s X M N BM BN sxb sxm sxk i (t.val * BK + e.val) = xs t i e) ∧
  (∀ t e, s.readMemTyped .int RMS ((t.val * BK + e.val) * srms) = some (rs t e).val) ∧
  (∀ t e j, s.readMemTyped .int W
    ((t.val * BK + e.val) * swk + colIndex s N BN j * swn) = some (ws t e j).val)

theorem finite_of_typed (s : BlockState) (r : RegionName) (o : Nat) (b : Byte)
    (h : s.readMemTyped .int r o = some b.val) : ByteLoads.FiniteAt s r o := by
  simpa [ByteLoads.FiniteAt, BlockState.readMemValue, h] using b.property

theorem view_of_typed (W RMS : RegionName) (s : BlockState) (r : RegionName)
    (o : Nat) (b : Byte) (hw : r = W ∨ r = RMS)
    (h : s.readMemTyped .int r o = some b.val) :
    (ByteLoads.view W RMS s).readMem r o = b.value := by
  have hv := ByteLoads.decoded_view W RMS s r o hw (finite_of_typed s r o b h)
  simp only [BlockState.readMemValue, h, b.decoded] at hv
  exact (Option.some.inj hv).symm

theorem inputs_bytes (s : BlockState) (X W RMS : RegionName)
    (M N K BM BN BK sxb sxm sxk swk swn srms : Nat)
    (xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (cdiv K BK) → Fin BK → Byte)
    (ws : Fin (cdiv K BK) → Fin BK → Fin BN → Byte)
    (h : Inputs s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms xs rs ws) :
    Bytes s W RMS N BN BK swk swn srms (cdiv K BK) := by
  refine ⟨?_, ?_⟩
  · intro t ht e j
    have hh := finite_of_typed _ _ _ _ (h.2.2 ⟨t, ht⟩ e j)
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hh
  · intro t ht e
    have hh := finite_of_typed _ _ _ _ (h.2.1 ⟨t, ht⟩ e)
    simpa [Nat.add_mul, Nat.add_comm] using hh

/-- The decoded-view recurrence equals the independent pure array formula. -/
theorem formula (s : BlockState) (X W RMS : RegionName)
    (M N K BM BN BK sxb sxm sxk swk swn srms : Nat) (EPS : ℝ)
    (xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (cdiv K BK) → Fin BK → Byte)
    (ws : Fin (cdiv K BK) → Fin BK → Fin BN → Byte)
    (hXW : X ≠ W) (hXR : X ≠ RMS)
    (h : Inputs s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms xs rs ws)
    (i : Fin BM) (j : Fin BN) :
    rmsSpec (ByteLoads.view W RMS s) X W RMS M N BM BN sxb sxm sxk swk swn srms
        BK (cdiv K BK) K EPS i j =
      PaddedRMSGemm.output K (cdiv K BK) BM BN BK EPS xs
        (fun t e => (rs t e).value) (fun t e j => (ws t e j).value) i j := by
  have hx : ∀ t i e,
      xElem (ByteLoads.view W RMS s) X M N BM BN sxb sxm sxk i (t.val * BK + e.val) = xs t i e := by
    intro t i e
    have he : xElem (ByteLoads.view W RMS s) X M N BM BN sxb sxm sxk i (t.val * BK + e.val) =
        xElem s X M N BM BN sxb sxm sxk i (t.val * BK + e.val) := by
      simp only [xElem, rowIndex, rowGlobal, ByteLoads.view, BlockState.readMem,
        hXW, hXR, false_or, ↓reduceIte]
    exact he.trans (h.1 t i e)
  have hr : ∀ t e, rmsElem (ByteLoads.view W RMS s) RMS srms (t.val * BK + e.val) = (rs t e).value := by
    intro t e
    exact view_of_typed W RMS s RMS _ _ (by simp) (h.2.1 t e)
  have hw : ∀ t e j, wElem (ByteLoads.view W RMS s) W N BN swk swn j (t.val * BK + e.val) = (ws t e j).value := by
    intro t e j
    exact view_of_typed W RMS s W _ _ (by simp) (h.2.2 t e j)
  simp only [rmsSpec, normSpec, accPartial, PaddedRMSGemm.output, PaddedRMSGemm.norm,
    PaddedRMSGemm.weightedSum, hx, hr, hw]

/-- Full source execution from independent finite byte arrays, with the
actual raw memory preserved outside active outputs. -/
theorem arrays (input : TypedSource.Storage) (outputFP8 variant : Bool)
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (cdiv K BK) → Fin BK → Byte)
    (ws : Fin (cdiv K BK) → Fin BK → Fin BN → Byte)
    (hu : ∀ rg o, s.undef rg o = 0) (hc : 0 < son) (hr : N * son ≤ som)
    (hXW : X ≠ W) (hXR : X ≠ RMS)
    (hinput : Inputs s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms xs rs ws) :
    ∃ t, exec (TypedSource.kernel input Bool.true outputFP8 variant X W RMS OUT M N K
        sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS).eraseDType s = some t ∧
      (∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        t.readMem OUT (outOffset s N BM BN sob som son idx) =
          PaddedRMSGemm.output K (cdiv K BK) BM BN BK EPS xs
            (fun t e => (rs t e).value) (fun t e j => (ws t e j).value) idx.1 idx.2.1) ∧
      (∀ r o, (r ≠ OUT ∨ ∀ idx : TileIndex [BM, BN], active s M N BM BN idx →
        o ≠ outOffset s N BM BN sob som son idx) → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, _hp, _hu, hv, hf⟩ := run input outputFP8 variant X W RMS OUT s M N K
    sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS hu hXW hXR
    (inputs_bytes s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms xs rs ws hinput)
    (Geometry.active_injective s M N BM BN sob som son hc hr)
  refine ⟨t, he, ?_, hf⟩
  intro idx hi
  rw [hv idx hi, formula s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms EPS
    xs rs ws hXW hXR hinput]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.ByteExecution
