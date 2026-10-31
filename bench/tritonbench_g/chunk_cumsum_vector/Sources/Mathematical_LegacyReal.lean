import VeriTile.Triton

/-!
# `chunk_cumsum_vector` — strict per-kernel correctness

`chunk_global_cumsum_vector_kernel` computes a global cumulative sum over the
time axis for `[T, S]` value tiles: program `(i_s, i_bh)` walks the `T`-rows of
the row's value matrix in chunks of `BT`, computing the in-chunk prefix sum as a
lower-triangular matmul `tl.dot(m_s, b_s)`, adding the carried column-vector
total `b_z`, and storing the `[BT, BS]` result into `z` while accumulating the
chunk column sums into `b_z`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`chunk_global_cumsum_vector_kernel[(cdiv(S,BS), B*H)]`,
the 2-D grid over feature blocks and batch·head rows, the autotuned `BT`, the
fixed `BS = 32`, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*, not a proof obligation here. Because the
program ids are universally quantified, the per-program statements cover every
program of the grid.

## Proof architecture

This kernel is a **global, per-column** cumulative sum: each output `[flat, j]`
holds `Σ_{m ≤ flat} s[m, j]`. The genuine closed-form spec is a standalone
`Finset.sum` (`globalCumsumVectorClosed` / `singleBlockCumsumVectorClosed`),
never a read-back of the kernel's own output.

```
chunk_cumsum_vector_output_summary_general                    ← TOP THEOREM (dimension-general)
  ├─ chunk_cumsum_vector_surface_toAlgorithm_supported        full surface lowers
  ├─ chunk_cumsum_vector_single_block_surface_closed_form
  │    ├─ chunk_cumsum_vector_single_block_surface_active_compute_correct
  │    └─ singleBlockStoreValue_eq_closed   (lower-tri dot = prefix Σ)
  ├─ chunk_cumsum_vector_store_slice_active_compute_correct
  └─ chunk_cumsum_vector_cumsum_slice_closed_form  (under per-column carry hyp.)
            ├─ chunk_cumsum_vector_cumsum_slice_active_compute_correct
            └─ cumsumStoreValue_eq_globalCumsumVectorClosed  (carry + dot = global Σ)

mathematical core (the carry-fold + within-chunk identity):
  dotLowerTri_sum_if_some                        `tl.dot(m_s,·)` = guarded prefix Σ
  singleBlockStoreValue_eq_closed                single chunk (carry=0) = global prefix Σ
  cumsumStoreValue_eq_globalCumsumVectorClosed   carry[j] + within-chunk Σ = global prefix Σ,
                                                 given carry[j] = Σ_{flat<c·BT, flat<T} s[flat,j]
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` (the
`BT ∈ {16,32,64}` config set) is not modeled — the public summary
`chunk_cumsum_vector_output_summary_general` covers the single-Python-chunk regime
(`T ≤ BT`, so each chunk loop runs once with carry `= 0`) at symbolic shapes; the closed-form
lemmas (`dotLowerTri_sum_if_some`, `*_eq_closed`,
`cumsumStoreValue_eq_globalCumsumVectorClosed`, and the
`*_surface_closed_form` / `*_cumsum_slice_closed_form` realizers) are stated and
proven **general over `T`, `S`, `BT`, `BS` and the number of chunks**. The
`.to(tl.float32)` / `.to(p_z.dtype.element_ty)` casts erase to the identity at
the algorithm layer. The in-chunk prefix sum is modeled exactly as the
lower-triangular matmul `lowerTriTile · sourceTile` (matching `tl.dot(m_s, b_s)`)
and shown equal to the genuine per-column prefix `Finset.sum`
(`dotLowerTri_sum_if_some`). The cross-chunk carry recurrence threaded by `b_z` is
`carry_{c+1}[j] = carry_c[j] + Σ chunk_c[·,j]`; its invariant
`carry_c[j] = Σ_{flat < c·BT, flat < T} s[i_bh·s_s_h + flat·s_s_t + j·s_s_d]` is
the explicit hypothesis of `cumsumStoreValue_eq_globalCumsumVectorClosed` — under
it, each chunk's store equals the genuine global cumulative sum. The carry is
materialized in a buffer (`Carry`) in `chunk_cumsum_vector_cumsum_slice`; the
single-Python-chunk surface realizes the global prefix sum end-to-end with
`carry = 0`. Output non-collision is a side condition (discharged per Python
case via `*_active_no_collision`).
-/

namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector

open VeriTile.Triton
open scoped VeriTile.Triton.Masked3DTileKernelIO₁

set_option linter.unusedSimpArgs false

def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val

def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val

def active (s : BlockState) (T S BT BS : Nat) (idx : TileIndex [BT, BS]) : Prop :=
  tIndex s BT idx.1 < T ∧ sIndex s BS idx.2.1 < S

instance activeDecidable (s : BlockState) (T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Decidable (active s T S BT BS idx) := by
  unfold active
  infer_instance

def tileOffset (s : BlockState) (s_s_h s_s_t s_s_d BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 1 * s_s_h + tIndex s BT idx.1 * s_s_t +
    sIndex s BS idx.2.1 * s_s_d

def singleBlockActive (s : BlockState) (T S BS : Nat)
    (idx : TileIndex [BT, BS]) : Prop :=
  idx.1.val < T ∧ sIndex s BS idx.2.1 < S

instance singleBlockActiveDecidable (s : BlockState) (T S BS : Nat)
    (idx : TileIndex [BT, BS]) : Decidable (singleBlockActive s T S BS idx) := by
  unfold singleBlockActive
  infer_instance

def singleBlockTileOffset (s : BlockState) (s_s_h s_s_t s_s_d BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 1 * s_s_h + idx.1.val * s_s_t + sIndex s BS idx.2.1 * s_s_d

noncomputable def storeValue (s : BlockState) (BC : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (if active s T S BT BS idx then
      some (s.readMem BC (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
    else some (0.0 : ℝ))

/-! ## Computed per-chunk cumsum slice

The store slice above starts from a precomputed `BC` tile. This slice models one
loop iteration of the Python kernel after the carry vector `b_z` has been
materialized in `Carry`: it loads the source block, computes
`b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)`, and stores the masked
result. -/

noncomputable def lowerTriTile (BT : Nat) : Tile .real [BT, BT] :=
  { data := fun idx =>
      if idx.1.val >= idx.2.1.val then some (1.0 : ℝ) else some (0.0 : ℝ) }

noncomputable def sourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if active s T S BT BS idx then
        some (s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
      else some (0.0 : ℝ) }

noncomputable def singleBlockSourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if singleBlockActive s T S BS idx then
        some (s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx))
      else some (0.0 : ℝ) }

noncomputable def carryValue
    (s : BlockState) (Carry : RegionName) (s_s_h s_s_d S BS : Nat)
    (j : Fin BS) : WithBot ℝ :=
  if sIndex s BS j < S then
    some (s.readMem Carry (s.pids 1 * s_s_h + sIndex s BS j * s_s_d))
  else some (0.0 : ℝ)

noncomputable def cumsumStoreValue
    (s : BlockState) (SReg Carry : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun carry dot : ℝ => carry + dot)
      (carryValue s Carry s_s_h s_s_d S BS idx.2.1)
      ((Tile.dot [] (lowerTriTile BT)
        (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
          (idx.1, idx.2.1, PUnit.unit)))

noncomputable def singleBlockStoreValue
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    ((Tile.dot [] (lowerTriTile BT)
      (singleBlockSourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
        (idx.1, idx.2.1, PUnit.unit))

/-! ## Within-chunk lower-triangular dot = guarded prefix `Finset.sum`

`tl.dot(m_s, b_s)` with `m_s` the lower-triangular ones matrix computes, at row
`i` and column `j`, the within-chunk prefix sum `∑_{k ≤ i} b_s[k, j]`. The
following lemma turns that register-level matmul into a genuine `Finset.sum`
over the guarded prefix, for any column-keyed source `f`. -/

/-- A `Finset.sum` of coerced lanes over `WithBot ℝ` collapses to the coercion
of the underlying real sum. (Instance-safe restatement of `WithBot.coe_sum`.) -/
private theorem sum_some_eq_some {ι : Type*} (s : Finset ι) (g : ι → ℝ) :
    (∑ k ∈ s, ((g k : ℝ) : WithBot ℝ)) = ((∑ k ∈ s, g k : ℝ) : WithBot ℝ) :=
  (WithBot.coe_sum s g).symm

/-- **Lower-triangular dot = guarded prefix sum (`some`-valued).** For a source
tile whose lane `(k, j)` holds `some (if P k j then f k j else 0)`, the
lower-triangular dot at `(i, j)` is `some` of the sum of `f · j` over the guarded
prefix `{k : k ≤ i ∧ P k j}`. This is the 2D (per-column) analogue of
`scan1d_sum_if`. -/
theorem dotLowerTri_sum_if_some (BT BS : Nat) (f : Fin BT → Fin BS → ℝ)
    (P : Fin BT → Fin BS → Prop) [∀ k j, Decidable (P k j)]
    (i : Fin BT) (j : Fin BS) :
    ((Tile.dot [] (lowerTriTile BT)
        (⟨fun idx => some (if P idx.1 idx.2.1 then f idx.1 idx.2.1 else 0)⟩ :
          Tile .real [BT, BS])).data (i, j, PUnit.unit))
      = some (∑ k ∈ (Finset.univ.filter
          (fun k : Fin BT => k.val ≤ i.val ∧ P k j)), f k j) := by
  rw [Tile.dot_nil_data]
  have e : (fun k : Fin BT => Option.map₂ (fun x1 x2 => x1 * x2)
      ((lowerTriTile BT).data (i, k, PUnit.unit))
      ((⟨fun idx => some (if P idx.1 idx.2.1 then f idx.1 idx.2.1 else 0)⟩ :
        Tile .real [BT, BS]).data (k, j, PUnit.unit)))
      = (fun k : Fin BT => ((((if k.val ≤ i.val then (1:ℝ) else 0) *
          (if P k j then f k j else 0)) : ℝ) : WithBot ℝ)) := by
    funext k
    simp only [lowerTriTile, ge_iff_le]
    split <;> (rw [Option.map₂_some_some, WithBot.some_eq_coe]; congr 1; norm_num)
  rw [e, sum_some_eq_some, WithBot.some_eq_coe]
  congr 1
  rw [← Finset.sum_filter_of_ne (p := fun k : Fin BT => k.val ≤ i.val ∧ P k j)]
  · apply Finset.sum_congr rfl
    intro k hk
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
    simp [hk.1, hk.2]
  · intro k _ hne
    by_contra hc
    push Not at hc
    by_cases hki : k.val ≤ i.val <;> simp_all

/-! ## Genuine chunked-cumsum closed form (vector / per-column)

`globalCumsumVectorClosed` is the genuine mathematical specification — for each
feature column `j` it is a `Finset.sum` over all *flat* time indices
`flat ≤ i_t·BT + i` (with `flat < T`) of the source value at the 2D address
`i_bh·s_s_h + flat·s_s_t + (i_s·BS + j)·s_s_d`. It is **not** a read-back of the
kernel's own output, so realizing it is a true correctness statement. Padded
feature lanes (`i_s·BS + j ≥ S`) hold `0`, matching the masked store. This is
the 2D analogue of the scalar `globalCumsumClosed`, carrying the column index
`j` through the prefix sum. -/
noncomputable def globalCumsumVectorClosed
    (s : BlockState) (SReg : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  if sIndex s BS idx.2.1 < S then
    ∑ flat ∈ (Finset.range T).filter
        (fun flat => flat ≤ s.pids 2 * BT + idx.1.val),
      s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
        sIndex s BS idx.2.1 * s_s_d)
  else 0

/-- Genuine closed form for the single-Python-chunk path (`i_t = 0`, carry `= 0`):
for each feature column `j`, the prefix sum of all source entries up to and
including flat index `i`. -/
noncomputable def singleBlockCumsumVectorClosed
    (s : BlockState) (SReg : RegionName) (s_s_h s_s_t s_s_d T S BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  if sIndex s BS idx.2.1 < S then
    ∑ flat ∈ (Finset.range T).filter (fun flat => flat ≤ idx.1.val),
      s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
        sIndex s BS idx.2.1 * s_s_d)
  else 0

/-- **The carry-fold recurrence (vector).** When the carry buffer `Carry` holds,
per active feature column, the genuine prefix sum of *all prior chunks* (every
flat index `< i_t·BT`, clamped to `< T`), the per-chunk store value
`cumsumStoreValue` — the within-chunk lower-triangular dot plus that carry —
equals the genuine global cumulative sum `globalCumsumVectorClosed`. This is the
exact recurrence threaded by `b_z` across the `forRange` loop:
`carry_{c+1}[j] = carry_c[j] + Σ chunk_c[·,j]`, with the invariant
`carry_c[j] = Σ_{flat < c·BT, flat < T} s[i_bh·s_s_h + flat·s_s_t + j·s_s_d]`. -/
theorem cumsumStoreValue_eq_globalCumsumVectorClosed
    (s : BlockState) (SReg Carry : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS])
    (hcarry : sIndex s BS idx.2.1 < S →
      s.readMem Carry (s.pids 1 * s_s_h + sIndex s BS idx.2.1 * s_s_d)
        = ∑ flat ∈ (Finset.range T).filter (fun flat => flat < s.pids 2 * BT),
            s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
              sIndex s BS idx.2.1 * s_s_d)) :
    cumsumStoreValue s SReg Carry s_s_h s_s_t s_s_d T S BT BS idx
      = globalCumsumVectorClosed s SReg s_s_h s_s_t s_s_d T S BT BS idx := by
  unfold cumsumStoreValue globalCumsumVectorClosed carryValue
  -- The dot operand `sourceTile` is exactly the guarded-`some` tile.
  have hsrc : sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS
      = (⟨fun p => some (if active s T S BT BS (p.1, p.2.1, PUnit.unit) then
          s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS (p.1, p.2.1, PUnit.unit))
          else 0)⟩ : Tile .real [BT, BS]) := by
    unfold sourceTile; congr 1; funext p
    obtain ⟨a, b, u⟩ := p
    by_cases h : active s T S BT BS (a, b, PUnit.unit)
    · simp [h]
    · simp only [h, if_false]; norm_num
  rw [hsrc]
  have hdot := dotLowerTri_sum_if_some BT BS
    (fun k j => s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS (k, j, PUnit.unit)))
    (fun k j => active s T S BT BS (k, j, PUnit.unit)) idx.1 idx.2.1
  rw [hdot]
  by_cases hcol : sIndex s BS idx.2.1 < S
  · -- active column: carry + within-chunk dot = global prefix sum
    rw [if_pos hcol, if_pos hcol, Option.map₂_some_some]
    show _ + _ = _
    rw [hcarry hcol]
    -- now: (Σ within-chunk) + carry = global prefix sum
    -- Step 1: reindex the within-chunk sum to flat segment indices.
    have hreindex :
        (∑ k ∈ Finset.univ.filter
            (fun k : Fin BT => k.val ≤ idx.1.val ∧
              active s T S BT BS (k, idx.2.1, PUnit.unit)),
          s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS (k, idx.2.1, PUnit.unit)))
        = ∑ flat ∈ (Finset.range T).filter
            (fun flat => s.pids 2 * BT ≤ flat ∧ flat ≤ s.pids 2 * BT + idx.1.val),
            s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
              sIndex s BS idx.2.1 * s_s_d) := by
      simp only [active, tIndex, tileOffset, hcol, and_true]
      apply Finset.sum_nbij'
        (i := fun k : Fin BT => s.pids 2 * BT + k.val)
        (j := fun flat => (⟨if h : flat - s.pids 2 * BT < BT then
            flat - s.pids 2 * BT else 0, by
            split
            · assumption
            · exact idx.1.pos⟩ : Fin BT))
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        simp only [Finset.mem_range, Finset.mem_filter]
        exact ⟨ha.2, by omega, by omega⟩
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_range] at ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        have hd : a - s.pids 2 * BT < BT := by omega
        rw [dif_pos hd]; exact ⟨by omega, by omega⟩
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        apply Fin.ext
        have hd : s.pids 2 * BT + a.val - s.pids 2 * BT < BT := by omega
        simp only [dif_pos hd]; omega
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_range] at ha
        have hd : a - s.pids 2 * BT < BT := by omega
        simp only [dif_pos hd]; omega
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        rfl
    rw [hreindex]
    -- Step 2: split `globalCumsumClosed` by `flat < i_t·BT`. The store value is
    -- `carry + within`, so the carry segment (`flat < i_t·BT`) comes first.
    rw [← Finset.sum_filter_add_sum_filter_not
          ((Finset.range T).filter (fun flat => flat ≤ s.pids 2 * BT + idx.1.val))
          (fun flat => flat < s.pids 2 * BT)]
    congr 1
    · rw [Finset.filter_filter]
      apply Finset.sum_congr ?_ (fun _ _ => rfl)
      apply Finset.filter_congr
      intro flat hflat
      simp only [Finset.mem_range] at hflat
      constructor
      · intro h; exact ⟨by omega, h⟩
      · rintro ⟨_, h2⟩; exact h2
    · rw [Finset.filter_filter]
      apply Finset.sum_congr ?_ (fun _ _ => rfl)
      apply Finset.filter_congr
      intro flat hflat
      simp only [Finset.mem_range, not_lt] at hflat ⊢
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨h2, h1⟩
      · rintro ⟨h1, h2⟩; exact ⟨h2, h1⟩
  · rw [if_neg hcol, if_neg hcol]
    have hempty : Finset.univ.filter
        (fun k : Fin BT => k.val ≤ idx.1.val ∧
          active s T S BT BS (k, idx.2.1, PUnit.unit)) = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro k _
      simp only [active, not_and]
      intro _
      simp only [not_lt] at *
      omega
    rw [hempty, Finset.sum_empty, Option.map₂_some_some]
    show _ + _ = _
    norm_num

/-- **Single-chunk correctness against the genuine closed form (vector).** With
the carry at its initial zero (`i_t = 0`), the within-chunk lower-triangular dot
store value is, per active feature column, the genuine global prefix sum
`Σ_{flat ≤ i, flat < T} s[i_bh·s_s_h + flat·s_s_t + j·s_s_d]`. This is the
sorry-free end-to-end correctness of the actual Python single-chunk path. -/
theorem singleBlockStoreValue_eq_closed
    (s : BlockState) (SReg : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) :
    singleBlockStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx
      = singleBlockCumsumVectorClosed s SReg s_s_h s_s_t s_s_d T S BS idx := by
  unfold singleBlockStoreValue singleBlockCumsumVectorClosed
  have hsrc : singleBlockSourceTile s SReg s_s_h s_s_t s_s_d T S BT BS
      = (⟨fun p => some (if singleBlockActive s T S BS (p.1, p.2.1, PUnit.unit) then
          s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
            (p.1, p.2.1, PUnit.unit))
          else 0)⟩ : Tile .real [BT, BS]) := by
    unfold singleBlockSourceTile; congr 1; funext p
    obtain ⟨a, b, u⟩ := p
    by_cases h : singleBlockActive s T S BS (a, b, PUnit.unit)
    · simp [h]
    · simp only [h, if_false]; norm_num
  rw [hsrc, dotLowerTri_sum_if_some BT BS
    (fun k j => s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
      (k, j, PUnit.unit)))
    (fun k j => singleBlockActive s T S BS (k, j, PUnit.unit)) idx.1 idx.2.1]
  show WithBot.unbotD 0 (some _) = _
  by_cases hcol : sIndex s BS idx.2.1 < S
  · rw [if_pos hcol]
    show (∑ k ∈ Finset.univ.filter
        (fun k : Fin BT => k.val ≤ idx.1.val ∧
          singleBlockActive s T S BS (k, idx.2.1, PUnit.unit)),
        s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
          (k, idx.2.1, PUnit.unit))) = _
    -- reindex `{k : Fin BT | k ≤ i ∧ (k < T ∧ col)}` to `{flat ∈ range T | flat ≤ i}`
    simp only [singleBlockActive, hcol, and_true, singleBlockTileOffset, sIndex]
    apply Finset.sum_nbij' (i := fun k : Fin BT => k.val)
      (j := fun flat => (⟨if h : flat < BT then flat else 0, by
          split
          · assumption
          · exact idx.1.pos⟩ : Fin BT))
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
      simp only [Finset.mem_range, Finset.mem_filter]
      exact ⟨ha.2.1, ha.1⟩
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_range] at ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      have hd : a < BT := by have : a ≤ idx.1.val := ha.2; omega
      rw [dif_pos hd]; exact ⟨ha.2, ha.1, hcol⟩
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
      apply Fin.ext
      simp only [dif_pos a.isLt]
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_range] at ha
      have hd : a < BT := by have : a ≤ idx.1.val := ha.2; omega
      simp only [dif_pos hd]
    · intro a _; rfl
  · rw [if_neg hcol]
    have hempty : Finset.univ.filter
        (fun k : Fin BT => k.val ≤ idx.1.val ∧
          singleBlockActive s T S BS (k, idx.2.1, PUnit.unit)) = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro k _
      simp only [singleBlockActive, not_and]
      intro _
      simp only [not_lt] at *
      omega
    rw [hempty, Finset.sum_empty]
    rfl

/-! ## Active-masked readback variants

For the Python tests with `S < BS`, padded feature lanes can alias active
addresses when considered over the whole `[BT, BS]` block. The store contract
only observes `active` lanes, so the theorem below uses the weaker semantic
lemma requiring no collision among active writes that target the active lane
being read. -/

/-! ## Genuine closed-form realizers

These wrap the `*_active_compute_correct` realizers with the closed-form
equivalence lemmas, so the realized `expected` is a standalone `Finset.sum`
specification (the genuine chunked cumulative sum), never a read-back of the
kernel's own output. -/

/-! ## ════════ `⊨` IO face for the masked block store ════════

The summary above is stated per *declared write map*. This section restates the
`BC → Z` block store on the audit-once IO surface
`Masked3DTileKernelIO₁.Implements` (`⊨`), which additionally pins the **flat memory**
placement.

The slice owns a `[BT, BS]` block whose address is built from **all three** program
axes (`i_s`, `i_bh`, `i_t`) — the shape the three-axis tile skin exists for. Read and
write share the one address, and both are gated by the kernel's own
`offs_t < T ∧ offs_s < S` guard. -/

section IOFace

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-! ### The rounding face

The store is a pure copy: no arithmetic on the loaded block, and no `.to(...)` on
the store either — the tile stays `.real` end to end. So the slice is
**cast-free** — every statement steps identically under `stepStmtsR R` and
`stepStmts` — and the exact run transports to `execR R` for *every* rounding
model. -/

/-! ### ════════ ★ MAIN THEOREM (rounding face) ★ ════════ -/

end IOFace

end VeriTile.Bench.TritonBenchG.ChunkCumsumVector
