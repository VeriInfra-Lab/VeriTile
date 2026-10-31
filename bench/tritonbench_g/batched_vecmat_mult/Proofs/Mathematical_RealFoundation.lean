import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Existing full-loop real execution, independent GEMV formula and streamed
flat-memory IO foundation. The .real source here is the precision-erased
mathematical program. Its identity rounding facts make no statement about
hardware floating point; current typed sources are in ../Kernels.lean.
The auxiliary one-row slices are retained only as supporting real results. -/

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `batched_vecmat_closed_form_correct` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + i.val

def outOffset (s : BlockState) (dim_n BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + i.val)

def aOffset (s : BlockState) (dim_k : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_k + j.val

def bOffset (s : BlockState) (dim_n dim_k BLOCK_N : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_n * dim_k + nIndex s BLOCK_N i * dim_k + j.val

noncomputable def vecmatProdTile
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    Tile .real [BLOCK_N, BLOCK_K] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let kj := (TileShape.dropInsertedIndex [BLOCK_K] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if kj.val < K then
          some (s.readMem A (aOffset s dim_k kj))
        else some (0.0 : ℝ))
        (if nIndex s BLOCK_N ni < N ∧ kj.val < K then
          some (s.readMem B (bOffset s dim_n dim_k BLOCK_N ni kj))
        else some (0.0 : ℝ)) }

@[standalone_spec] noncomputable def vecmatSpec
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (vecmatProdTile s A B dim_n dim_k N K BLOCK_N BLOCK_K)).data
        (i, PUnit.unit))

def kIndex (s : BlockState) (BLOCK_K : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 2 * BLOCK_K + j.val

def aOffsetK (s : BlockState) (dim_k BLOCK_K : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_k + kIndex s BLOCK_K j

def bOffsetK (s : BlockState) (dim_n dim_k BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_n * dim_k + nIndex s BLOCK_N i * dim_k +
    kIndex s BLOCK_K j

noncomputable def vecmatProdTileK
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    Tile .real [BLOCK_N, BLOCK_K] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let kj := (TileShape.dropInsertedIndex [BLOCK_K] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if kIndex s BLOCK_K kj < K then
          some (s.readMem A (aOffsetK s dim_k BLOCK_K kj))
        else some (0.0 : ℝ))
        (if nIndex s BLOCK_N ni < N ∧ kIndex s BLOCK_K kj < K then
          some (s.readMem B (bOffsetK s dim_n dim_k BLOCK_N BLOCK_K ni kj))
        else some (0.0 : ℝ)) }

noncomputable def vecmatSpecK
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (vecmatProdTileK s A B dim_n dim_k N K BLOCK_N BLOCK_K)).data
        (i, PUnit.unit))

noncomputable def vecmatAccumSpecK
    (s : BlockState) (AccPre A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun old delta => old + delta)
      (some (s.readMem AccPre (outOffset s dim_n BLOCK_N i)))
      ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
        (vecmatProdTileK s A B dim_n dim_k N K BLOCK_N BLOCK_K)).data
          (i, PUnit.unit)))

def withKIndex (s : BlockState) (kIdx : Nat) : BlockState :=
  { s with pids := fun ax => if ax = 2 then kIdx else s.pids ax }

@[standalone_spec] noncomputable def vecmatConstKAccumSpec
    (s : BlockState) (AccPre A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K kIdx : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  vecmatAccumSpecK (withKIndex s kIdx) AccPre A B dim_n dim_k N K
    BLOCK_N BLOCK_K i

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def blockOutOffset (s : BlockState) (dim_n BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  mIndex s BLOCK_M idx.1 * dim_n + nIndex s BLOCK_N idx.2.1

@[standalone_spec] noncomputable def blockOutputStoreSpec
    (s : BlockState) (VecmatPre : RegionName) (dim_n BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  s.readMem VecmatPre (blockOutOffset s dim_n BLOCK_M BLOCK_N idx)

/-! ## Genuine GEMV closed-form correctness over the full dynamic K-loop

`for k_index in range(k_blocks)` (with `k_blocks = dim_k // block_k` a runtime
register) lowers to `Stmt.forRangeDyn`. The lemmas below build a closed-form
characterization of the kernel's `output` against the independent batched
vector-matrix reference `out[m,n] = Σ_{kk < dim_k} A[m,kk] · B[m,n,kk]`. -/

/-- Master invariant principle for the *dynamic* `forRangeDyn` loop. Resolves
the runtime bounds via `evalOp`, then drives the body by the caller's invariant
`P`, exposing the loop-exit counter `final` (with `stop ≤ final`). -/
theorem forRangeDyn_inv
    {idx : RegName} {startOp stopOp stepOp : Op .nat []}
    {start stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (hStart : evalOp startOp s_init = some (Tile.scalar start))
    (hStop : evalOp stopOp s_init = some (Tile.scalar stop))
    (hStepOp : evalOp stepOp s_init = some (Tile.scalar step))
    (hstep : step ≠ 0)
    (h_init : P start s_init)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s') :
    ∃ final s_final,
      stepStmt (.forRangeDyn idx startOp stopOp stepOp body) s_init = some s_final ∧
      stop ≤ final ∧ P final s_final := by
  obtain ⟨final, s_final, h_aux, hfinal, hP⟩ :=
    forRangeAux_inv hstep h_step start s_init h_init
  refine ⟨final, s_final, ?_, hfinal, hP⟩
  rw [stepForRangeAux.forRangeDyn_unfold, hStart, hStop, hStepOp]
  simp only [Option.bind]
  exact h_aux

/-! ### Genuine batched vector-matrix (GEMV) spec -/

/-- Global output row of tile lane `i`: `m_index · BLOCK_M + i`. -/
def gRow (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

/-- Global output column of tile lane `j`: `n_index · BLOCK_N + j`. -/
def gCol (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val

/-- `A[m, kk] = readMem A (gRow i · dim_k + kk)` (the kernel's row-major A layout). -/
noncomputable def aElem (s : BlockState) (A : RegionName) (dim_k BLOCK_M : Nat)
    (i : Fin BLOCK_M) (kk : Nat) : ℝ :=
  s.readMem A (gRow s BLOCK_M i * dim_k + kk)

/-- `B[m, n, kk] = readMem B (gRow i · dim_n · dim_k + gCol j · dim_k + kk)`
(the kernel's row-major B layout). -/
noncomputable def bElem (s : BlockState) (B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) (kk : Nat) : ℝ :=
  s.readMem B (gRow s BLOCK_M i * dim_n * dim_k + gCol s BLOCK_N j * dim_k + kk)

/-- **Genuine GEMV spec**: `out[i,j] = Σ_{kk < BLOCK_K·numKBlocks} A[i,kk] · B[i,j,kk]`. -/
@[standalone_spec] noncomputable def gemvSpec (s : BlockState) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) : ℝ :=
  (Finset.range (BLOCK_K * numKBlocks)).sum
    (fun kk => aElem s A dim_k BLOCK_M i kk
      * bElem s B dim_n dim_k BLOCK_M BLOCK_N i j kk)

/-- Partial GEMV accumulator after `c` K-blocks: `Σ_{kk < c·BLOCK_K} A·B`. -/
noncomputable def gemvPartial (s : BlockState) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) (c : Nat) : ℝ :=
  (Finset.range (c * BLOCK_K)).sum
    (fun kk => aElem s A dim_k BLOCK_M i kk
      * bElem s B dim_n dim_k BLOCK_M BLOCK_N i j kk)

/-- One-block step of the partial accumulator: the new block contributes the
dot over the `BLOCK_K` keys `c·BLOCK_K + e`. -/
theorem gemvPartial_succ (s : BlockState) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) (c : Nat) :
    gemvPartial s A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K i j (c + 1)
      = gemvPartial s A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            aElem s A dim_k BLOCK_M i (c * BLOCK_K + e.val)
              * bElem s B dim_n dim_k BLOCK_M BLOCK_N i j (c * BLOCK_K + e.val)) := by
  unfold gemvPartial
  have h : (c + 1) * BLOCK_K = c * BLOCK_K + BLOCK_K := by ring
  rw [h, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_range fun e => aElem s A dim_k BLOCK_M i (c * BLOCK_K + e)
        * bElem s B dim_n dim_k BLOCK_M BLOCK_N i j (c * BLOCK_K + e)]

/-! ### Body decomposition (prefix ++ for-loop ++ store) -/

/-! ### Loop invariant -/

/-- **Loop invariant** (counter `c` = number of K-blocks done so far).

After `c` K-blocks: program ids and `mem`/`undef` fixed; the `m_index` /
`n_index` / `output_tile` registers seeded; and `vecmat` equals the partial
GEMV accumulator `gemvPartial … c`. -/
noncomputable def vecmatInvariant
    (A B output : RegionName) (s0 : BlockState)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧
  (s.regs .nat [] "m_index" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "n_index" = some (Tile.scalar (s0.pids 1))) ∧
  (s.regs .nat [BLOCK_M, BLOCK_N] "output_tile" = some
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        (s0.pids 0 * BLOCK_M + idx.1.val) * dim_n + (s0.pids 1 * BLOCK_N + idx.2.1.val)⟩) ∧
  (s.regs .real [BLOCK_M, BLOCK_N] "vecmat" = some
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (gemvPartial s0 A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 c)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-! ### evalOp helpers for the loop-body statements -/

/-- `a_tile` eval: cell `(i,k) = (gRow i)·dim_k + (c·BLOCK_K + k)`. -/
theorem vecmat_atile_eval (s : BlockState) (dim_k BLOCK_M BLOCK_K mi ki : Nat)
    (hm : s.regs .nat [] "m_index" = some (Tile.scalar mi))
    (hk : s.regs .nat [] "k_index" = some (Tile.scalar ki)) :
    evalOp
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
              (Op.arange BLOCK_M)))
          (Op.constNat dim_k))
        (Op.expandDim ⟨0, by simp⟩
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
            (Op.arange BLOCK_K)))) s
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_K] =>
          (mi * BLOCK_M + idx.1.val) * dim_k + (ki * BLOCK_K + idx.2.1.val)⟩ : Tile .nat [BLOCK_M, BLOCK_K]) := by
  simp only [evalOp.eq_def, hm, hk, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.expandDim, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul]

/-- `b_tile` eval: cell `(j,i,k) = (gRow i)·dim_n·dim_k + (gCol j)·dim_k + (c·BLOCK_K + k)`. -/
theorem vecmat_btile_eval (s : BlockState) (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K mi ni ki : Nat)
    (hm : s.regs .nat [] "m_index" = some (Tile.scalar mi))
    (hn : s.regs .nat [] "n_index" = some (Tile.scalar ni))
    (hk : s.regs .nat [] "k_index" = some (Tile.scalar ki)) :
    evalOp
      (Op.add NumericDType.nat Broadcast.nil.consL.consR.consR
        (Op.add NumericDType.nat Broadcast.nil.consSame.consR.consL
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨2, by simp⟩
                (Op.expandDim ⟨0, by simp⟩
                  (Op.add NumericDType.nat Broadcast.scalarL
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
                    (Op.arange BLOCK_M))))
              (Op.constNat dim_n))
            (Op.constNat dim_k))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨2, by simp⟩
              (Op.expandDim ⟨1, by simp⟩
                (Op.add NumericDType.nat Broadcast.scalarL
                  (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "n_index") (Op.constNat BLOCK_N))
                  (Op.arange BLOCK_N))))
            (Op.constNat dim_k)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.expandDim ⟨0, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
              (Op.arange BLOCK_K))))) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_M, BLOCK_K] =>
          (mi * BLOCK_M + idx.2.1.val) * dim_n * dim_k
            + (ni * BLOCK_N + idx.1.val) * dim_k
            + (ki * BLOCK_K + idx.2.2.1.val)⟩ : Tile .nat [BLOCK_N, BLOCK_M, BLOCK_K]) := by
  simp only [evalOp.eq_def, hm, hn, hk, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.expandDim, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex, TileShape.insertAxis,
    NumericDType.add, NumericDType.mul]

/-- No-mask region load eval: reads `readMem` at each offset. -/
theorem vecmat_load_eval {shape : TileShape} (s : BlockState) (R : RegionName) (name : RegName)
    (offs : Tile .nat shape) (h : s.regs .nat shape name = some offs) :
    evalOp (Op.load TileDType.real (MemAccess.region R (Op.ref TileDType.nat shape name)) MaskOpt.none) s
      = some (⟨fun i => some (s.readMem R (offs.data i))⟩ : Tile .real shape) := by
  simp only [evalOp_load_region_none, evalOp_ref, h, Option.bind]
  refine congrArg some ?_
  ext i
  simp [BlockState.readMemValue_real]

/-- **`vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))` eval.**

Given the loaded `expanded_a`/`b` tiles whose lanes are the genuine `A`/`B`
elements at K-offset `c·BLOCK_K + k`, and `vecmat` holding the partial sum after
`c` blocks, the update yields the partial sum after `c + 1` blocks. -/
theorem vecmat_update_eval (s s0 : BlockState) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K c : Nat)
    (aT : Tile .real [BLOCK_M, BLOCK_K]) (bT : Tile .real [BLOCK_N, BLOCK_M, BLOCK_K])
    (hvm : s.regs .real [BLOCK_M, BLOCK_N] "vecmat" = some
        ⟨fun idx => some (gemvPartial s0 A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 c)⟩)
    (hea : s.regs .real [1, BLOCK_M, BLOCK_K] "expanded_a" = some (Tile.expandDim ⟨0, by simp⟩ aT))
    (hb : s.regs .real [BLOCK_N, BLOCK_M, BLOCK_K] "b" = some bT)
    (haval : ∀ (i : Fin BLOCK_M) (k : Fin BLOCK_K),
        aT.data (i, k, PUnit.unit) = some (aElem s0 A dim_k BLOCK_M i (c * BLOCK_K + k.val)))
    (hbval : ∀ (j : Fin BLOCK_N) (i : Fin BLOCK_M) (k : Fin BLOCK_K),
        bT.data (j, i, k, PUnit.unit)
          = some (bElem s0 B dim_n dim_k BLOCK_M BLOCK_N i j (c * BLOCK_K + k.val))) :
    evalOp
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame
        (Op.ref TileDType.real [BLOCK_M, BLOCK_N] "vecmat")
        (Op.transpose (batch := []) (M := BLOCK_N) (N := BLOCK_M)
          (Op.reduceSum ⟨2, by simp⟩ Bool.false
            (Op.mul NumericDType.real Broadcast.nil.consSame.consSame.consL
              (Op.ref TileDType.real [1, BLOCK_M, BLOCK_K] "expanded_a")
              (Op.ref TileDType.real [BLOCK_N, BLOCK_M, BLOCK_K] "b"))))) s
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          some (gemvPartial s0 A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 (c + 1))⟩
            : Tile .real [BLOCK_M, BLOCK_N]) := by
  simp only [evalOp.eq_def, hvm, hea, hb, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  -- reduce the per-lane value to vecmat[i,j] + Σ_k a[i,k]·b[j,i,k]
  simp only [Tile.bop, Tile.transpose, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul, WithBot.realAdd]
  rw [Tile.reduceSum_false, Tile.reduceSumDrop_data]
  simp only [TileShape.insertAxisIndex, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.dropInsertedIndex, haval, hbval, WithBot.realMul, Option.map₂,
    Option.bind, Option.map, WithBot.some_eq_coe]
  -- the block sum is now `∑ ↑(aElem · bElem)`; collapse to `↑(∑ …)` and fold
  rw [← WithBot.coe_sum, gemvPartial_succ]
  rfl

/-! ### preLoop and postLoop -/

/-- `output_tile` eval: cell `(i,j) = (gRow i)·dim_n + (gCol j)`. -/
theorem vecmat_otile_eval (s : BlockState) (dim_n BLOCK_M BLOCK_N mi ni : Nat)
    (hm : s.regs .nat [] "m_index" = some (Tile.scalar mi))
    (hn : s.regs .nat [] "n_index" = some (Tile.scalar ni)) :
    evalOp
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
              (Op.arange BLOCK_M)))
          (Op.constNat dim_n))
        (Op.expandDim ⟨0, by simp⟩
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "n_index") (Op.constNat BLOCK_N))
            (Op.arange BLOCK_N)))) s
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          (mi * BLOCK_M + idx.1.val) * dim_n + (ni * BLOCK_N + idx.2.1.val)⟩ : Tile .nat [BLOCK_M, BLOCK_N]) := by
  simp only [evalOp.eq_def, hm, hn, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.expandDim, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul]

/-- The output store address for tile lane `(i,j)`: `gRow i · dim_n + gCol j`. -/
def vecmatOutOffset (s : BlockState) (dim_n BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  gRow s BLOCK_M idx.1 * dim_n + gCol s BLOCK_N idx.2.1

/-! ### Composition: full exec closed form -/

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-! ## Python test-shape summary

`test_vecmat` runs `M = N = K = 128`, `block_m = 16`, `block_n = 32`,
`block_k = 64`, so `numKBlocks = 128 / 64 = 2`. -/

end Correct_without_Rounding

/-! ## The `⊨[R]` streaming headline (wave-5 S1 fold genre)

Everything below is purely additive; the exact surface above is untouched.
Structure of the `execR R` story: this kernel has **zero rounding events** —
the two unmasked region loads, the `expanded_a * b` product, the axis-2
`tl.sum`, the `tl.trans` and the accumulator add are all at `.real`, and the
terminal store is an unmasked `.real` region store, which `stepStmtR`
delegates to the exact `writeMemTyped` (a `.real` write is not a
quantization). The 5-statement prologue and the whole `forRangeDyn` K-loop
are therefore cast-free and collapse verbatim onto the exact stepper
(`stepForRangeAuxR_castFree`), reusing the proven `vecmat_preLoop` /
`vecmat_step` / `forRangeDyn_inv` invariant stack unchanged; only the single
store statement is re-proved on the `R` side (`vecmat_postLoopR`), where the
`.real` write coincides with the `writeMemAsR R .real` rounded write because
`R.storeValue .real` **is** the exact store demotion
(`RoundingModel.storeValue_real`). The skin's readback contract then carries
`R.round .real`, which is the identity by the model's defining `round_real`. -/

open scoped VeriTile.Triton.StreamMasked2DKernelIO₂

/-! ### Body decomposition names and cast-free collapses -/

/-- The `a_tile` address op is cast-free. -/
private theorem evalR_atile (R : RoundingModel) (dim_k BLOCK_M BLOCK_K : Nat) (s : BlockState) :
    evalOpR R
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
              (Op.arange BLOCK_M)))
          (Op.constNat dim_k))
        (Op.expandDim ⟨0, by simp⟩
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
            (Op.arange BLOCK_K)))) s
      = evalOp
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
              (Op.arange BLOCK_M)))
          (Op.constNat dim_k))
        (Op.expandDim ⟨0, by simp⟩
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
            (Op.arange BLOCK_K)))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `b_tile` address op is cast-free. -/
private theorem evalR_btile (R : RoundingModel)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) (s : BlockState) :
    evalOpR R
      (Op.add NumericDType.nat Broadcast.nil.consL.consR.consR
        (Op.add NumericDType.nat Broadcast.nil.consSame.consR.consL
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨2, by simp⟩
                (Op.expandDim ⟨0, by simp⟩
                  (Op.add NumericDType.nat Broadcast.scalarL
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
                    (Op.arange BLOCK_M))))
              (Op.constNat dim_n))
            (Op.constNat dim_k))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨2, by simp⟩
              (Op.expandDim ⟨1, by simp⟩
                (Op.add NumericDType.nat Broadcast.scalarL
                  (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "n_index") (Op.constNat BLOCK_N))
                  (Op.arange BLOCK_N))))
            (Op.constNat dim_k)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.expandDim ⟨0, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
              (Op.arange BLOCK_K))))) s
      = evalOp
      (Op.add NumericDType.nat Broadcast.nil.consL.consR.consR
        (Op.add NumericDType.nat Broadcast.nil.consSame.consR.consL
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨2, by simp⟩
                (Op.expandDim ⟨0, by simp⟩
                  (Op.add NumericDType.nat Broadcast.scalarL
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
                    (Op.arange BLOCK_M))))
              (Op.constNat dim_n))
            (Op.constNat dim_k))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨2, by simp⟩
              (Op.expandDim ⟨1, by simp⟩
                (Op.add NumericDType.nat Broadcast.scalarL
                  (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "n_index") (Op.constNat BLOCK_N))
                  (Op.arange BLOCK_N))))
            (Op.constNat dim_k)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.expandDim ⟨0, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
              (Op.arange BLOCK_K))))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-! ### The tail under `execR R` -/

/-- A `.real`-typed rounded write **is** the `writeMemAsR R .real` write:
`writeMemTypedR` delegates `.real` to the exact `writeMemTyped`, and
`RoundingModel.storeValue_real` says `R.storeValue .real` is exactly that
store demotion. -/
private theorem writeMemTypedR_real_eq (R : RoundingModel) (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier TileDType.real) :
    s.writeMemTypedR R .real region offset v
      = s.writeMemAsR R .real region offset v := by
  show s.writeMemTyped .real region offset v = _
  simp only [BlockState.writeMemTyped, BlockState.writeMemAs, BlockState.writeMemAsR,
    RoundingModel.storeValue_real]

/-- Tag-exact readback of a stored `.real` cell through `readMemAs .real`. -/
private theorem readMemAs_real_of_cell {s : BlockState} {region : RegionName}
    {offset : Nat} {x : ℝ}
    (h : s.mem region offset
      = MemCell.of FloatDType.real.toTileDType (FloatDType.real.ofReal x)) :
    s.readMemAs .real region offset = FloatDType.real.ofReal x := by
  simp [BlockState.readMemAs, h, FloatDType.storeValue, FloatDType.ofReal]

/-! ### Safety-walk invariant (weak shape half of `vecmatInvariant`) -/

/-- Safety-walk loop invariant: the *shape* half of `vecmatInvariant`
(`m_index` / `n_index` / `output_tile` seeded, *some* accumulator tile) with
no `undef` / `mem` / value pins, and no dependence on the loop counter — the
body recomputes both address tiles from `m_index` / `n_index` / `k_index`
every iteration. Needed because the `⊨[R]` skin's `hts` obligation quantifies
over arbitrary launch states, so the safety walk cannot assume the
clean-`undef` precondition that `vecmat_preLoop`'s full invariant needs. -/
private def vecmatSafeInv (s0 : BlockState) (dim_n BLOCK_M BLOCK_N : Nat)
    (s : BlockState) : Prop :=
  (s.regs .nat [] "m_index" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "n_index" = some (Tile.scalar (s0.pids 1))) ∧
  (s.regs .nat [BLOCK_M, BLOCK_N] "output_tile" = some
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        (s0.pids 0 * BLOCK_M + idx.1.val) * dim_n + (s0.pids 1 * BLOCK_N + idx.2.1.val)⟩) ∧
  (∃ vT : Tile .real [BLOCK_M, BLOCK_N], s.regs .real [BLOCK_M, BLOCK_N] "vecmat" = some vT)

/-! ### The lane bridges -/

/-- The `A`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[BLOCK_M, BLOCK_N]` output tile) paired with `e`
over the `[BLOCK_M, BLOCK_K]` per-step `A`-tile. -/
def aLane (BLOCK_M BLOCK_N BLOCK_K : Nat) (l : Fin (BLOCK_M * BLOCK_N)) (e : Fin BLOCK_K) :
    Fin (BLOCK_M * BLOCK_K) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)

/-- The `B`-stream lane feeding output lane `l` at inner key `e`: the
`[BLOCK_N, BLOCK_M, BLOCK_K]` cell `(j, i, e)` flattened row-major through
two nested `Lane2D` bridges — the `(j, i)` pair first (into
`Fin (BLOCK_N * BLOCK_M)`), then the key. -/
def bLane (BLOCK_M BLOCK_N BLOCK_K : Nat) (l : Fin (BLOCK_M * BLOCK_N)) (e : Fin BLOCK_K) :
    Fin (BLOCK_N * BLOCK_M * BLOCK_K) :=
  Lane2D.encode
    (Lane2D.encode ((Lane2D.decode l).2.1, (Lane2D.decode l).1, PUnit.unit), e, PUnit.unit)

/-! ### The safety walk -/

/-! ### IO signature and spec bridge -/

/-- Under the two stream pins, `gemvSpec` at the decoded output lane **is**
the skin-level double fold `∑ t, ∑ e, xs · ys` (`gemmSum_blocks` +
address-identity of the windows with the kernel's `a_tile` / `b_tile`
shapes). -/
private theorem gemvSpec_eq_streamSum (A B : RegionName) (s₀ : BlockState)
    (dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (xs : Fin numKBlocks → Fin (BLOCK_M * BLOCK_K) → ℝ)
    (ys : Fin numKBlocks → Fin (BLOCK_N * BLOCK_M * BLOCK_K) → ℝ)
    (hx : ∀ (t : Fin numKBlocks) (l : Fin (BLOCK_M * BLOCK_K)),
      s₀.readMem A ((s₀.pids 0 * BLOCK_M + l.val / BLOCK_K) * (BLOCK_K * numKBlocks)
          + (t.val * BLOCK_K + l.val % BLOCK_K))
        = xs t l)
    (hy : ∀ (t : Fin numKBlocks) (l : Fin (BLOCK_N * BLOCK_M * BLOCK_K)),
      s₀.readMem B ((s₀.pids 0 * BLOCK_M + l.val / BLOCK_K % BLOCK_M) * dim_n
            * (BLOCK_K * numKBlocks)
          + (s₀.pids 1 * BLOCK_N + l.val / BLOCK_K / BLOCK_M) * (BLOCK_K * numKBlocks)
          + (t.val * BLOCK_K + l.val % BLOCK_K))
        = ys t l)
    (l : Fin (BLOCK_M * BLOCK_N)) :
    gemvSpec s₀ A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K numKBlocks
        (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = ∑ t : Fin numKBlocks, ∑ e : Fin BLOCK_K,
          xs t (aLane BLOCK_M BLOCK_N BLOCK_K l e) * ys t (bLane BLOCK_M BLOCK_N BLOCK_K l e) := by
  rw [show gemvSpec s₀ A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K numKBlocks
        (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = gemmSum
          (fun kk => aElem s₀ A (BLOCK_K * numKBlocks) BLOCK_M (Lane2D.decode l).1 kk)
          (fun kk => bElem s₀ B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N
            (Lane2D.decode l).1 (Lane2D.decode l).2.1 kk)
          (numKBlocks * BLOCK_K) from by
    unfold gemvSpec gemmSum
    rw [Nat.mul_comm BLOCK_K numKBlocks]]
  rw [gemmSum_blocks]
  refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => ?_
  have hxa : aElem s₀ A (BLOCK_K * numKBlocks) BLOCK_M (Lane2D.decode l).1
      (t.val * BLOCK_K + e.val) = xs t (aLane BLOCK_M BLOCK_N BLOCK_K l e) := by
    rw [← hx t (aLane BLOCK_M BLOCK_N BLOCK_K l e)]
    simp only [aLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  have hyb : bElem s₀ B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N
      (Lane2D.decode l).1 (Lane2D.decode l).2.1 (t.val * BLOCK_K + e.val)
      = ys t (bLane BLOCK_M BLOCK_N BLOCK_K l e) := by
    rw [← hy t (bLane BLOCK_M BLOCK_N BLOCK_K l e)]
    simp only [bLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  rw [hxa, hyb]

/-! ### The headline -/

end VeriTile.Bench.TritonBenchG.BatchedVecmatMult
