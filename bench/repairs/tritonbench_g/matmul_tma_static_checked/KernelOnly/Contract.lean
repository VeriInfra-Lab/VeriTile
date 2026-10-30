import bench.tritonbench_g.matmul_tma.Proofs.Layout
import Init.Data.Nat.Power2.Lemmas

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked
open Host

/-- Public integer shape/configuration domain checked before allocation.
Device capability and resource availability are separate applicability facts. -/
structure InputContract (h : Shape) (cfg : Config) : Prop where
  rows : h.rows.isPowerOfTwo
  columns : h.columns.isPowerOfTwo
  inner : h.inner.isPowerOfTwo
  inner_min : 16 ≤ h.inner
  warps : cfg.numWarps.isPowerOfTwo
  ctas : cfg.numCtas.isPowerOfTwo

/-- Literal bit-test used by the Python positive-power-of-two guards. -/
def powerGuard (n : Nat) : Bool := decide (n ≠ 0 ∧ n &&& (n-1) = 0)

def checkInputs (h : Shape) (cfg : Config) : Except String Unit :=
  if powerGuard h.rows then
    if powerGuard h.columns then
      if h.inner < 16 then .error "K must be a power of two and at least 16"
      else if powerGuard h.inner then
        if powerGuard cfg.numWarps then
          if powerGuard cfg.numCtas then .ok ()
          else .error "NUM_CTAS must be a positive power of two"
        else .error "NUM_WARPS must be a positive power of two"
      else .error "K must be a power of two and at least 16"
    else .error "N must be a positive power of two"
  else .error "M must be a positive power of two"

theorem powerGuard_iff (n : Nat) : powerGuard n = Bool.true ↔ n.isPowerOfTwo := by
  simp only [powerGuard,decide_eq_true_eq,Nat.ne_zero_and_sub_one_eq_zero_iff_isPowerOfTwo]

theorem checkInputs_iff (h : Shape) (cfg : Config) :
    checkInputs h cfg = .ok () ↔ InputContract h cfg := by
  have contract : InputContract h cfg ↔ h.rows.isPowerOfTwo ∧ h.columns.isPowerOfTwo ∧
      16 ≤ h.inner ∧ h.inner.isPowerOfTwo ∧ cfg.numWarps.isPowerOfTwo ∧ cfg.numCtas.isPowerOfTwo := by
    constructor
    · intro d; exact ⟨d.rows,d.columns,d.inner_min,d.inner,d.warps,d.ctas⟩
    · rintro ⟨a,b,c,d,e,f⟩; exact ⟨a,b,d,c,e,f⟩
  rw [contract]
  simp [checkInputs,powerGuard_iff,ite_eq_iff,and_assoc]

theorem inner_multiple (h : Shape) (cfg : Config) (d : InputContract h cfg) : 16 ∣ h.inner := by
  obtain ⟨e,he⟩ := d.inner
  have hk := d.inner_min
  have emin : 4 ≤ e := by
    by_contra no
    have small : e < 4 := by omega
    interval_cases e <;> norm_num at he <;> omega
  rw [he]
  change 2^4 ∣ 2^e
  exact pow_dvd_pow 2 emin

#axiomsClean checkInputs_iff
#axiomsClean inner_multiple
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked
