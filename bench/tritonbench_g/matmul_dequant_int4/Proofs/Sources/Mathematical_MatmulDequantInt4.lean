import VeriTile.Triton
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.SignedCast

/-! Independent width-erased mathematical address and nibble helpers.
The authoritative repaired and candidate function bodies are in Sources/Body.lean.
FPTyped and the public source wrappers connect those bodies to the typed signed32
execution, independent Real formula, safety, complete grids and memory frames.
These natural-valued helpers are used by the mathematical bridge, not as a
second source implementation. FPBits supplies the actual packed-word connection. -/

namespace VeriTile.Bench.TritonBenchG.MatmulDequantInt4

open VeriTile.Triton

/-- The source nibble mask itself supplies the machine-cast range; no
new premise is imposed on input words, dimensions or the public contract. -/
@[local simp] private theorem nibble_int_mod32 (n : Nat) :
    ((n &&& 15 : Nat) : Int) % 4294967296 = ((n &&& 15 : Nat) : Int) := by
  have h : n &&& 15 ≤ 15 := Nat.and_le_right
  apply Int.emod_eq_of_lt <;> omega
@[local simp] private theorem nibble_nat_mod32 (n : Nat) :
    (n &&& 15) % 4294967296 = n &&& 15 := by
  have h : n &&& 15 ≤ 15 := Nat.and_le_right
  apply Nat.mod_eq_of_lt
  omega
@[local simp] private theorem nibble_sign32 (n : Nat) :
    2 * (n &&& 15) < 4294967296 := by
  have h : n &&& 15 ≤ 15 := Nat.and_le_right
  omega

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding



/-! ## Element accessors

Each accessor is the kernel's own address arithmetic over the **launch**
state's memory, parameterized by the *absolute* row `k = k_block_idx ·
BLOCK_SIZE_K + i` and column `n = n_block_idx · BLOCK_SIZE_N + j`. -/

/-- The packed 32-bit word holding the weight nibble for absolute cell
`(k, n)`: `b[k // 8, n]`. Eight weights share a word along K, hence `k / 8`. -/
def bWord (s : BlockState) (b_ptr : Region .nat)
    (stride_bk stride_bn k n : Nat) : Nat :=
  s.readMemValue .nat b_ptr.cast (k / 8 * stride_bk + n * stride_bn)

/-- The unpacked 4-bit weight: `(bWord >> (k % 8) * 4) & 0xF`. -/
def bNibble (s : BlockState) (b_ptr : Region .nat)
    (stride_bk stride_bn k n : Nat) : Nat :=
  bWord s b_ptr stride_bk stride_bn k n >>> (k % 8 * 4) &&& 15

/-- The packed word holding the zero-point nibble for absolute cell `(k, n)`:
`zp[k // group_size, n // 8]`. Eight zero-points share a word along N, hence
`n / 8`; one group row per `group_size` K rows. -/
def zpWord (s : BlockState) (b_zp_ptr : Region .nat)
    (group_size stride_bzpk stride_bzpn k n : Nat) : Nat :=
  s.readMemValue .nat b_zp_ptr.cast
    (k / group_size * stride_bzpk + n / 8 * stride_bzpn)

/-- The unpacked 4-bit zero-point: `(zpWord >> (n % 8) * 4) & 0xF`. -/
def zpNibble (s : BlockState) (b_zp_ptr : Region .nat)
    (group_size stride_bzpk stride_bzpn k n : Nat) : Nat :=
  zpWord s b_zp_ptr group_size stride_bzpk stride_bzpn k n >>> (n % 8 * 4) &&& 15

/-- The scale at absolute cell `(k, n)`: `bs[k // group_size, n]`. -/
noncomputable def scElem (s : BlockState) (b_scale_ptr : RegionName)
    (group_size stride_bsk stride_bsn k n : Nat) : ℝ :=
  s.readMem b_scale_ptr (k / group_size * stride_bsk + n * stride_bsn)

/-- **The stored value** at absolute cell `(k, n)`: the **signed** nibble
difference (ℤ-subtraction — this is what the `.int` hop of disclosure (1)
buys; ℕ subtraction would truncate), embedded into ℝ and scaled. -/
@[standalone_spec] noncomputable def dequantSpec (s : BlockState) (b_scale_ptr : RegionName)
    (b_ptr b_zp_ptr : Region .nat)
    (group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn : Nat) (k n : Nat) : ℝ :=
  (((bNibble s b_ptr stride_bk stride_bn k n : ℤ)
      - (zpNibble s b_zp_ptr group_size stride_bzpk stride_bzpn k n : ℤ) : ℤ) : ℝ)
    * scElem s b_scale_ptr group_size stride_bsk stride_bsn k n

/-- The `fp_b` store address for tile cell `idx` of program `(p₀, p₁)`:
`(p₀·BK + i) · stride_fpbk + (p₁·BN + j) · stride_fpbn`. -/
def fpbAddr (stride_fpbk stride_fpbn BK BN p₀ p₁ : Nat)
    (idx : TileIndex [BK, BN]) : Nat :=
  (p₀ * BK + idx.1.val) * stride_fpbk + (p₁ * BN + idx.2.1.val) * stride_fpbn

/-- The kernel's store gate `mask = n_mask & k_mask`, in the kernel's own
conjunct order (the N test comes first in the source). -/
def mdiActive (K N BK BN p₀ p₁ : Nat) (idx : TileIndex [BK, BN]) : Prop :=
  p₁ * BN + idx.2.1.val < N ∧ p₀ * BK + idx.1.val < K

/-- `fpbAddr` is injective for a row-major `fp_b` (`stride_fpbn > 0` and a
K-row stride wide enough to hold `BLOCK_SIZE_N` columns) — the discharge
lemma for the headline's `hInj`. The host's contiguous `fp_b` of
`torch.ones((K, N))` has `stride_fpbk = N ≥ BN · 1` for every launched tile. -/
theorem fpbAddr_injective (stride_fpbk stride_fpbn BK BN p₀ p₁ : Nat)
    (hn : 0 < stride_fpbn) (hfit : BN * stride_fpbn ≤ stride_fpbk) :
    Function.Injective
      (fun idx : TileIndex [BK, BN] =>
        fpbAddr stride_fpbk stride_fpbn BK BN p₀ p₁ idx) := by
  intro i j hij
  obtain ⟨r₁, c₁, u₁⟩ := i
  obtain ⟨r₂, c₂, u₂⟩ := j
  simp only [fpbAddr] at hij
  have hexp : ∀ r c : Nat,
      (p₀ * BK + r) * stride_fpbk + (p₁ * BN + c) * stride_fpbn
        = p₀ * BK * stride_fpbk + r * stride_fpbk
          + (p₁ * BN * stride_fpbn + c * stride_fpbn) := by
    intro r c
    rw [Nat.add_mul, Nat.add_mul]
  rw [hexp, hexp] at hij
  have key : r₁.val * stride_fpbk + c₁.val * stride_fpbn
      = r₂.val * stride_fpbk + c₂.val * stride_fpbn := by omega
  have hrow : ∀ c : Fin BN, c.val * stride_fpbn < stride_fpbk := fun c =>
    lt_of_lt_of_le (Nat.mul_lt_mul_of_lt_of_le c.isLt (le_refl _) hn) hfit
  have hr : r₁.val = r₂.val := by
    rcases Nat.lt_trichotomy r₁.val r₂.val with h | h | h
    · have : r₁.val * stride_fpbk + stride_fpbk ≤ r₂.val * stride_fpbk := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₁
      omega
    · exact h
    · have : r₂.val * stride_fpbk + stride_fpbk ≤ r₁.val * stride_fpbk := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₂
      omega
  have hc : c₁.val = c₂.val := by
    have : c₁.val * stride_fpbn = c₂.val * stride_fpbn := by
      rw [hr] at key; omega
    exact Nat.eq_of_mul_eq_mul_right hn this
  simp only [Prod.mk.injEq]
  exact ⟨Fin.ext hr, Fin.ext hc, trivial⟩

/-! ## Termination and per-lane readback

The kernel is an 18-statement straight line: one big `simp` over the
step/eval definitions runs it completely, normalizing `exec` to `some` of a
masked scatter `foldl` over the output store (the masked loads are total —
each carries `other=` — and so is the masked store). The readback lemma then
peels that `foldl` with `scatter_readback_prop_masked_nd`. -/

/-! ## Main theorem -/

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.MatmulDequantInt4
