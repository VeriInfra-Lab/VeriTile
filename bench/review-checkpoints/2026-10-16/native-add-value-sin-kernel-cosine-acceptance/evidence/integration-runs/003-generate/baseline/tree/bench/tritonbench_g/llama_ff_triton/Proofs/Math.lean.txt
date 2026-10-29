import VeriTile.Triton
import VeriTile.Triton.Math.PaddedRMSGemm
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.Mathematics
open VeriTile.Triton

/-- The source visits all BK lanes of every ceiling-counted block. -/
def steps (K BK : Nat) : Nat := cdiv K BK

/-- Exact RMS moment of every loaded lane, including the unmasked K padding.
The source divides this padded sum by the logical K, not the padded extent. -/
noncomputable def norm (K BK BM : Nat) (eps : ℝ)
    (xs : Fin (steps K BK) → Fin BM → Fin BK → ℝ) (i : Fin BM) : ℝ :=
  PaddedRMSGemm.norm K (steps K BK) BM BK eps xs i

/-- Independent gated-GEMM sum over every source-loaded block and lane. -/
noncomputable def weightedSum (K BK BM BN : Nat)
    (xs : Fin (steps K BK) → Fin BM → Fin BK → ℝ)
    (rms : Fin (steps K BK) → Fin BK → ℝ)
    (weights : Fin (steps K BK) → Fin BK → Fin BN → ℝ)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  PaddedRMSGemm.weightedSum (steps K BK) BM BN BK xs rms weights i j

/-- The actual source formula on independently supplied input arrays. Padding
is ordinary loaded data; no synthetic zero-fill or logical-K truncation occurs. -/
noncomputable def output (K BK BM BN : Nat) (eps : ℝ)
    (xs : Fin (steps K BK) → Fin BM → Fin BK → ℝ)
    (rms : Fin (steps K BK) → Fin BK → ℝ)
    (w1 w3 : Fin (steps K BK) → Fin BK → Fin BN → ℝ)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  let n := norm K BK BM eps xs i
  let s1 := n * weightedSum K BK BM BN xs rms w1 i j
  let s3 := n * weightedSum K BK BM BN xs rms w3 i j
  (s1 * Real.sigmoid s1) * s3

/-- A real-valued byte input carries an actual finite decoder result. This
subtype has no NaN/Inf case and uses no default numerical value. -/
def Byte := { n : Int // Float8E5Bits.isInt8 n = Bool.true ∧
  (Float8E5Bits.ofInt n).decodeRat.isSome = Bool.true }

noncomputable def Byte.value (b : Byte) : ℝ :=
  (((Float8E5Bits.ofInt b.val).decodeRat.get b.property.2 : ℚ) : ℝ)

theorem Byte.decoded (b : Byte) :
    ((Float8E5Bits.ofInt b.val).decodeRat.map (fun q : ℚ => (q : ℝ))) = some b.value := by
  cases h : (Float8E5Bits.ofInt b.val).decodeRat with
  | none => have := b.property.2; simp [h] at this
  | some q => simp [Byte.value, h]

/-- Concrete geometry witnesses why a logical-K bound is insufficient: K=3,
BK=2 visits the fourth K lane. This is a source behavior, not masked padding. -/
theorem partial_block_reads_padding : steps 3 2 = 2 ∧ 3 < steps 3 2 * 2 := by
  decide

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.Mathematics
