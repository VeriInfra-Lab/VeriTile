import VeriTile.Triton.Math.RMSNorm

namespace VeriTile.Triton.PaddedRMSGemm

/-- RMS scale over the supplied blocks, with an explicit logical divisor.
Every array element participates, including any source-loaded padding. -/
noncomputable def norm (K blocks BM BK : Nat) (eps : ℝ)
    (xs : Fin blocks → Fin BM → Fin BK → ℝ) (i : Fin BM) : ℝ :=
  1 / Real.sqrt ((∑ t, ∑ e, (xs t i e) ^ 2) / (K : ℝ) + eps)

/-- Independently supplied input, channel scale and matrix weights determine
the complete weighted matrix product. No execution or memory object occurs. -/
noncomputable def weightedSum (blocks BM BN BK : Nat)
    (xs : Fin blocks → Fin BM → Fin BK → ℝ)
    (rs : Fin blocks → Fin BK → ℝ)
    (ws : Fin blocks → Fin BK → Fin BN → ℝ)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  ∑ t, ∑ e, xs t i e * rs t e * ws t e j

/-- A row's RMS scale multiplies its weighted matrix product. The number of
loaded blocks and the logical RMS divisor are separate parameters. -/
noncomputable def output (K blocks BM BN BK : Nat) (eps : ℝ)
    (xs : Fin blocks → Fin BM → Fin BK → ℝ)
    (rs : Fin blocks → Fin BK → ℝ)
    (ws : Fin blocks → Fin BK → Fin BN → ℝ)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  norm K blocks BM BK eps xs i * weightedSum blocks BM BN BK xs rs ws i j

end VeriTile.Triton.PaddedRMSGemm
