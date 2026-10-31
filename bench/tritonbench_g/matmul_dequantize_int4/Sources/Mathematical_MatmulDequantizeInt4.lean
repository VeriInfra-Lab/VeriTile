import VeriTile.Triton
import VeriTile.Meta.StatementAudit
import VeriTile.Meta.StatementAudit

/-!
# `matmul_dequantize_int4` — strict per-kernel correctness

`matmul_dequantize_int4.py`'s `matmul4_kernel` is a GPTQ-style dequantizing GEMM:
`C = A · dequant(qweight)`, where `qweight` holds **eight 4-bit weights packed per
32-bit word** along the K axis, and `qzeros` holds eight 4-bit zero-points packed
per word along the N axis. Each output block is accumulated over the K axis; on
every step the packed weights are unpacked with a shift and a nibble mask, scaled,
and shifted by the group's zero-point.

One program owns one `(pid_m, pid_n)` output block, chosen by the standard
group-swizzled `pid` decomposition (`GROUP_SIZE_M` rows of blocks at a time).

## The two dequantization channels

The packed tensors are `torch.IntTensor`, and both are modelled as `Region .nat`:
the DSL's bitwise operators (`>>`, `&`) are defined on the `.nat` channel only.
This is a statement about the *container*, not the extracted values —
`(x >> 4i) & 0xF` selects the same nibble whether the shift is logical or
arithmetic, and every extracted nibble is in `[0, 15]`, hence non-negative. The
unpacked values cross to `ℝ` immediately at `* scales`, so no integer step ever
needs to go negative.

## Proof map

```
matmul_dequantize_int4_exec_genuine             the headline
├─ mdq_body_eq                31 statements by `rfl`
├─ mdqPreLoopScalars_run      12 statements: pid swizzle + three `tl.cdiv`s
├─ mdqPreLoopTiles_run        → `mdqInv … 0`
│  ├─ mdqPreLoopIndex_run     10 statements: offsets, 4 pointer tiles, mask, shifters
│  └─ mdq_groupPreload_run    the `if NO_GROUPS` pre-load, at group row 0
├─ mdqLoop_collapse           `forRangeDyn_inv` over `mdqInv`
│  └─ mdqLoopBody_run         8 statements: `mdqInv i → mdqInv (i+1)`
│     ├─ mdq_groupBranch_run  the `if not NO_GROUPS` reload, at group row `groupRow`
│     ├─ mdqBDequantTile_eq   the two dequant statements
│     └─ mdqAccTile_dot_succ  `accumulator += tl.dot(a, b)`
└─ mdqPostLoop_run            6 statements: output coords, mask, masked `.ptr` store
   └─ mdq_store_props         scatter readback
cAddr_injective                                 discharges the headline's `hInj`
```

The stored value is `accSpec`, built bottom-up from the kernel's own masked
accessors (`aElem`, `bWord`/`bNibble`, `scalesElem`, `zerosWord`/`zerosNibble`) over
the **launch** state's memory. The output region `c_ptr` is never read back into a
spec, so no part of the trust path is self-referential.

## Modeling boundary

The source's `(b).to(a.dtype)` remains an explicit `.real → .real`
`Op.castFloat` in the algorithm AST and is evaluated through the identity
cast of this model. Arithmetic is over `ℝ` (not bit-accurate IEEE float);
`@triton.autotune` and the
host launch (the 1-D grid `cdiv(M,BM)·cdiv(N,BN)`, the block sizes, and
`NO_GROUPS = (group_size == K)`) are the *trusted boundary*. Every dimension,
stride and block size stays a symbolic parameter.

Two spelling notes, per `documents/bench/MAIN_THEOREM_CONVENTIONS.md`, both surface syntax
rather than semantics:

* integer literals inside index arithmetic are written `$(n)`, since a bare
  literal is inferred `.real` by the DSL's expression typing. The source's `0xF`
  is written `$(15)` for the same reason — same value, decimal spelling.
* the source's `bits = 4` / `infearure_per_bits = 8` are ordinary assignments and
  are transcribed as such (they are `tl.constexpr`-free Python locals, so they
  lower to real statements, not to inlined constants).

## Faithfulness note worth flagging

The source computes `c = accumulator.to(c_ptr.dtype.element_ty)` and then stores
**`accumulator`**, not `c` — so `c` is a dead binding. That is transcribed as
written; the spec below is therefore about the `float32` accumulator.

## Correctness-surface status

Correctness-surface blocker: the `*_exec_genuine` headline(s) in this file
state correctness as an inline exec-existential
(`∃ sF, exec … = some sF ∧ <output readback>`) rather than through a named
`⊨` (`KernelIO`) or `ComputeCorrect.Realizes` surface. Termination and every
stored output cell are proved; the **frame** — nothing outside the write set
changes — is not part of the statement. Lifting this to `⊨` therefore needs
the frame proof, not a rename. Registered in `proof_blockers.md`.
-/

namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4

open VeriTile.Triton

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! ## Kernel surface (faithful transcription) -/

/-! ## The group-swizzled block coordinates

`pid` is decomposed exactly as the source decomposes it, so the spec below is
stated in the same coordinates the kernel computes rather than in a re-derived
form. -/

/-- `num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)`. -/
def numPidM (M BM : Nat) : Nat := (M + BM - 1) / BM

/-- `num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)`. -/
def numPidN (N BN : Nat) : Nat := (N + BN - 1) / BN

/-- `num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)` — the K-loop trip count. -/
def numPidK (K BK : Nat) : Nat := (K + BK - 1) / BK

/-- `num_pid_in_group = GROUP_SIZE_M * num_pid_n`. -/
def numPidInGroup (N BN GM : Nat) : Nat := GM * numPidN N BN

/-- `group_id = pid // num_pid_in_group`. -/
def groupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / numPidInGroup N BN GM

/-- `first_pid_m = group_id * GROUP_SIZE_M`. -/
def firstPidM (s : BlockState) (N BN GM : Nat) : Nat := groupId s N BN GM * GM

/-- `group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)`. -/
def groupSizeM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (numPidM M BM - firstPidM s N BN GM) GM

/-- `pid_m = first_pid_m + (pid % group_size_m)`. -/
def pidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  firstPidM s N BN GM + s.pids 0 % groupSizeM s M N BM BN GM

/-- `pid_n = (pid % num_pid_in_group) // group_size_m`. -/
def pidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % numPidInGroup N BN GM / groupSizeM s M N BM BN GM

/-! ## Masked element accessors

Each is the kernel's own address arithmetic, guarded by the mask the matching load
carries. All read the **launch** state's memory. -/

/-- `a[r, k*BK + e]` at K step `k`. Masked on the row axis only — that is the
kernel's `a_mask = offs_am[:, None] < M` with `other=0.0`. There is deliberately
no K-axis mask: the source documents `K % BLOCK_SIZE_K == 0` as a precondition,
and transcribing a mask it does not have would be unfaithful. -/
noncomputable def aElem (s : BlockState) (a : RegionName)
    (M stride_am stride_ak BM BK pm : Nat) (r k e : Nat) : ℝ :=
  if pm * BM + r < M then
    s.readMem a ((pm * BM + r) * stride_am + (e + k * BK) * stride_ak)
  else 0

/-- The packed 32-bit word holding the weight nibble for `(k, e, c)`. Unmasked,
matching the source's bare `tl.load(b_ptrs)`. Eight weights share a word along K,
hence the `e / 8`. -/
def bWord (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn : Nat) (k e c : Nat) : Nat :=
  s.readMemValue .nat b
    ((e / 8 + k * (BK / 8)) * stride_bk + (pn * BN + c) * stride_bn)

/-- The unpacked 4-bit weight: `(word >> (e % 8) * 4) & 0xF`. -/
def bNibble (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn : Nat) (k e c : Nat) : Nat :=
  bWord s b stride_bk stride_bn BN BK pn k e c >>> (e % 8 * 4) &&& 15

/-- The group row read at K step `k`. Under `NO_GROUPS` the scales and zeros are
loaded once before the loop **from the base pointers**, i.e. at row `0`; otherwise
each step reloads at `k // (groupsize // BLOCK_SIZE_K)`. -/
def groupRow (NO_GROUPS : Bool) (groupsize BK k : Nat) : Nat :=
  if NO_GROUPS then 0 else k / (groupsize / BK)

/-- `scales[g, pn*BN + c]`. -/
noncomputable def scalesElem (s : BlockState) (scales : RegionName)
    (stride_scales_g stride_scales_n BN pn : Nat) (g c : Nat) : ℝ :=
  s.readMem scales ((pn * BN + c) * stride_scales_n + g * stride_scales_g)

/-- The packed word holding the zero-point nibble for column `pn*BN + c`. Eight
zero-points share a word along N, hence the `/ 8`. -/
def zerosWord (s : BlockState) (zeros : Region .nat)
    (stride_zeros_g stride_zeros_n BN pn : Nat) (g c : Nat) : Nat :=
  s.readMemValue .nat zeros
    ((pn * BN + c) / 8 * stride_zeros_n + g * stride_zeros_g)

/-- The unpacked 4-bit zero-point. -/
def zerosNibble (s : BlockState) (zeros : Region .nat)
    (stride_zeros_g stride_zeros_n BN pn : Nat) (g c : Nat) : Nat :=
  zerosWord s zeros stride_zeros_g stride_zeros_n BN pn g c
    >>> ((pn * BN + c) % 8 * 4) &&& 15

/-- `zeros` after the kernel's `zeros = zeros * scales`: the **scaled**
zero-point. Both branches apply this before the loop body uses it. -/
noncomputable def zeroScaled (s : BlockState) (scales : RegionName)
    (zeros : Region .nat)
    (stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN pn : Nat)
    (g c : Nat) : ℝ :=
  (zerosNibble s zeros stride_zeros_g stride_zeros_n BN pn g c : ℝ)
    * scalesElem s scales stride_scales_g stride_scales_n BN pn g c

/-- `b` after unpack, scale and zero-point shift at K step `k`:
`b * scales - zeros`, where `zeros` is already scaled. -/
noncomputable def bDequant (s : BlockState) (scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (groupsize stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BN BK pn : Nat) (k e c : Nat) : ℝ :=
  (bNibble s b stride_bk stride_bn BN BK pn k e c : ℝ)
      * scalesElem s scales stride_scales_g stride_scales_n BN pn
          (groupRow NO_GROUPS groupsize BK k) c
    - zeroScaled s scales zeros stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BN pn (groupRow NO_GROUPS groupsize BK k) c

/-! ## The accumulator

`tl.dot(a, b)` over the block's K extent, summed across the `num_pid_k` steps. -/

/-- One K step's contribution to output cell `(r, c)`. -/
noncomputable def accStep (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) (k r c : Nat) : ℝ :=
  ∑ e : Fin BK,
    aElem s a M stride_am stride_ak BM BK pm r k e.val
      * bDequant s scales b zeros NO_GROUPS groupsize stride_bk stride_bn
          stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
          BN BK pn k e.val c

/-- **The stored value.** `accumulator` after all `num_pid_k` K steps. (The source
also computes `c = accumulator.to(...)` but stores `accumulator`, so this is what
lands in memory.) -/
@[standalone_spec] noncomputable def accSpec (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) (r c : Nat) : ℝ :=
  ∑ k : Fin (numPidK K BK),
    accStep s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BM BN BK pm pn k.val r c

/-- The `C` store address for output cell `(r, c)`. -/
def cAddr (stride_cm stride_cn BM BN pm pn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * (pm * BM + idx.1.val) + stride_cn * (pn * BN + idx.2.1.val)

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by `rfl`
rather than assumed. Lowerings worth naming because they are not guessable from
the source text:

* `//` and `%` lower to `Op.floorDiv` / `Op.mod` on `IntegralDType.nat`, **not** to
  `Op.div` — `Op.div .nat` is what `tl.cdiv` expands to;
* `min(a, b)` is `Op.where (Op.lt …) a b` (there is no `Op.min`);
* a pointer expression `R + off` is `Op.ptrAdd Broadcast.scalarL (Op.ptrBase R) off`;
* **the rank-broadcast load mask** `a_mask : [BM, 1]` used on a `[BM, BK]` load is
  wrapped by the DSL in `Op.remap … Broadcast.leftIndex`, and the `other=0.0`
  makes it a `MaskOpt.maskOther` against `Op.broadcast (Op.const 0.0) [BM, BK]`;
* `nat * real` inserts `Op.natToReal` on the nat side. -/

/-! ## Pointer tiles

A `.ptr` tile carries one `(region, offset)` pair per lane, and a `.ptr` load is
unconditionally in bounds — the mask is the only gate. `a_ptrs` and `b_ptrs` are
advanced once per K step, so each is a function of the step `k`, and the two
`mdq*Ptrs_succ` lemmas are what lets the invariant carry across an iteration. -/

/-- `a_ptrs` lane `(r, e)` at K step `k`, in the accumulated form the kernel
actually reaches: the initial offset plus `k` advances of `BK * stride_ak`. -/
def mdqAAddr (stride_am stride_ak BM BK pm k : Nat) (idx : TileIndex [BM, BK]) : Nat :=
  (pm * BM + idx.1.val) * stride_am + idx.2.1.val * stride_ak + k * (BK * stride_ak)

/-- `b_ptrs` lane `(e, c)` at K step `k`. Eight weights share a word along K, so a
step advances by `(BK / 8) * stride_bk`. -/
def mdqBAddr (stride_bk stride_bn BN BK pn k : Nat) (idx : TileIndex [BK, BN]) : Nat :=
  idx.1.val / 8 * stride_bk + (pn * BN + idx.2.1.val) * stride_bn
    + k * (BK / 8 * stride_bk)

noncomputable def mdqAPtrs (a : RegionName) (stride_am stride_ak BM BK pm k : Nat) :
    Tile .ptr [BM, BK] :=
  ⟨fun idx => (a, mdqAAddr stride_am stride_ak BM BK pm k idx)⟩

noncomputable def mdqBPtrs (b : Region .nat)
    (stride_bk stride_bn BN BK pn k : Nat) : Tile .ptr [BK, BN] :=
  ⟨fun idx => (Region.cast b, mdqBAddr stride_bk stride_bn BN BK pn k idx)⟩

/-- One `a_ptrs += BLOCK_SIZE_K * stride_ak` advance. -/
theorem mdqAPtrs_succ (a : RegionName) (stride_am stride_ak BM BK pm k : Nat) :
    Tile.ptrAdd Broadcast.scalarR (mdqAPtrs a stride_am stride_ak BM BK pm k)
        (Tile.scalar (BK * stride_ak))
      = mdqAPtrs a stride_am stride_ak BM BK pm (k + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, mdqAPtrs, mdqAAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- One `b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk` advance. -/
theorem mdqBPtrs_succ (b : Region .nat) (stride_bk stride_bn BN BK pn k : Nat) :
    Tile.ptrAdd Broadcast.scalarR (mdqBPtrs b stride_bk stride_bn BN BK pn k)
        (Tile.scalar (BK / 8 * stride_bk))
      = mdqBPtrs b stride_bk stride_bn BN BK pn (k + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, mdqBPtrs, mdqBAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- The `a_ptrs` address agrees with `aElem`'s: the kernel's accumulated
`e*stride_ak + k*(BK*stride_ak)` is `aElem`'s `(e + k*BK)*stride_ak`. -/
theorem mdqAAddr_eq (stride_am stride_ak BM BK pm k : Nat) (idx : TileIndex [BM, BK]) :
    mdqAAddr stride_am stride_ak BM BK pm k idx
      = (pm * BM + idx.1.val) * stride_am + (idx.2.1.val + k * BK) * stride_ak := by
  simp only [mdqAAddr]
  ring

/-- The `b_ptrs` address agrees with `bWord`'s. -/
theorem mdqBAddr_eq (stride_bk stride_bn BN BK pn k : Nat) (idx : TileIndex [BK, BN]) :
    mdqBAddr stride_bk stride_bn BN BK pn k idx
      = (idx.1.val / 8 + k * (BK / 8)) * stride_bk
        + (pn * BN + idx.2.1.val) * stride_bn := by
  simp only [mdqBAddr]
  ring

/-! ## The loop body's value tiles

Each is named in the form the eval recipes will produce, with a separate `_data`
bridge to the spec accessor — the same split that kept `chunk_bwd_dqkg`'s address
arithmetic out of its step proofs. -/

/-- `a_mask` as the prologue leaves it: a `[BM, 1]` bool tile, `true` exactly on
in-region rows. -/
def mdqAMask (M BM pm : Nat) : Tile .bool [BM, 1] :=
  ⟨fun idx => decide (pm * BM + idx.1.val < M)⟩

/-- `shifter[e] = (e % 8) * 4` — `offs_k` is `arange BK`, so lane `e` holds `e`. -/
def mdqShifter (BK : Nat) : Tile .nat [BK] := ⟨fun idx => idx.1.val % 8 * 4⟩

/-- `zeros_shifter[c] = ((pn*BN + c) % 8) * 4`. -/
def mdqZerosShifter (BN pn : Nat) : Tile .nat [BN] :=
  ⟨fun idx => (pn * BN + idx.1.val) % 8 * 4⟩

/-- `a` at K step `k`. In-region rows read memory; the rest read the load's
`other=0.0`. The mask is rank-broadcast from `[BM, 1]`, so it does not depend on
the K lane — which is exactly why there is no K-axis boundary handling. -/
noncomputable def mdqATile (s : BlockState) (a : RegionName)
    (M stride_am stride_ak BM BK pm k : Nat) : Tile .real [BM, BK] :=
  ⟨fun idx => if pm * BM + idx.1.val < M then
      some (s.readMem a (mdqAAddr stride_am stride_ak BM BK pm k idx)) else some 0⟩

/-- The loaded packed weight words at K step `k`. Unmasked. -/
noncomputable def mdqBRaw (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn k : Nat) : Tile .nat [BK, BN] :=
  ⟨fun idx => s.readMemValue .nat b (mdqBAddr stride_bk stride_bn BN BK pn k idx)⟩

/-- `b` after the nibble extraction. -/
noncomputable def mdqBNibbleTile (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn k : Nat) : Tile .nat [BK, BN] :=
  ⟨fun idx => bNibble s b stride_bk stride_bn BN BK pn k idx.1.val idx.2.1.val⟩

/-- `accumulator` after `i` K steps: the partial sum the invariant carries. -/
noncomputable def mdqAccTile (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) (i : Nat) : Tile .real [BM, BN] :=
  ⟨fun idx => some (∑ j : Fin i,
      accStep s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
        stride_bk stride_bn stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BM BN BK pm pn j.val idx.1.val idx.2.1.val)⟩

/-- At `i = 0` the accumulator is the zero tile `tl.zeros` produces. -/
theorem mdqAccTile_zero (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) :
    mdqAccTile s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
        stride_bk stride_bn stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BM BN BK pm pn 0
      = (⟨fun _ => some 0⟩ : Tile .real [BM, BN]) := by
  apply Tile.ext
  intro idx
  simp [mdqAccTile]

/-- At `i = numPidK` the accumulator is `accSpec` — the stored value. -/
theorem mdqAccTile_full (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) (idx : TileIndex [BM, BN]) :
    (mdqAccTile s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
        stride_bk stride_bn stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BM BN BK pm pn (numPidK K BK)).data idx
      = some (accSpec s a scales b zeros NO_GROUPS M K groupsize stride_am
          stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
          stride_zeros_g stride_zeros_n BM BN BK pm pn idx.1.val idx.2.1.val) := by
  rfl

/-- **The invariant's accumulator step.** Extending the partial sum by one K step
adds that step's `tl.dot` contribution — the algebraic content of the fold. -/
theorem mdqAccTile_succ (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn i : Nat) (idx : TileIndex [BM, BN]) :
    (mdqAccTile s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
        stride_bk stride_bn stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BM BN BK pm pn (i + 1)).data idx
      = some ((∑ j : Fin i,
            accStep s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
              stride_bk stride_bn stride_scales_g stride_scales_n
              stride_zeros_g stride_zeros_n BM BN BK pm pn j.val
              idx.1.val idx.2.1.val)
          + accStep s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
              stride_bk stride_bn stride_scales_g stride_scales_n
              stride_zeros_g stride_zeros_n BM BN BK pm pn i
              idx.1.val idx.2.1.val) := by
  simp only [mdqAccTile]
  exact congrArg some (Fin.sum_univ_castSucc _)

/-! ### Bridges from the tiles to the spec accessors -/

theorem mdqATile_data (s : BlockState) (a : RegionName)
    (M stride_am stride_ak BM BK pm k : Nat) (idx : TileIndex [BM, BK]) :
    (mdqATile s a M stride_am stride_ak BM BK pm k).data idx
      = some (aElem s a M stride_am stride_ak BM BK pm idx.1.val k idx.2.1.val) := by
  simp only [mdqATile, aElem, mdqAAddr_eq]
  split <;> rfl

theorem mdqBRaw_data (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn k : Nat) (idx : TileIndex [BK, BN]) :
    (mdqBRaw s b stride_bk stride_bn BN BK pn k).data idx
      = bWord s b stride_bk stride_bn BN BK pn k idx.1.val idx.2.1.val := by
  simp only [mdqBRaw, bWord, mdqBAddr_eq]

/-- The nibble tile is `bitAnd 15` over `shiftRight`, applied to the raw words —
i.e. exactly the two compiled statements, read back through `bNibble`. -/
theorem mdqBNibbleTile_eq (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn k : Nat) :
    mdqBNibbleTile s b stride_bk stride_bn BN BK pn k
      = Tile.bop (· &&& ·) Broadcast.scalarR
          (Tile.bop (· >>> ·) (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (mdqBRaw s b stride_bk stride_bn BN BK pn k)
            (Tile.expandDim ⟨1, by simp⟩ (mdqShifter BK)))
          (Tile.scalar 15) := by
  apply Tile.ext
  intro idx
  simp only [mdqBNibbleTile, bNibble, Tile.bop_data, Tile.scalar,
    Tile.expandDim_data, mdqShifter, TileShape.dropInsertedIndex,
    Broadcast.leftIndex, Broadcast.rightIndex, mdqBRaw_data]

/-! ## Per-statement eval recipes

Private copies, since bench ports never import each other. The `.ptr` family is
new for this port — `chunk_bwd_dqkg` used block pointers, whose loads carry a
boundary check; a `.ptr` load is unconditionally in bounds, so the mask is the
only gate and there is no `ok` branch to discharge. -/

/-- Unmasked `.ptr` load: every lane reads its own `(region, offset)`. -/
private theorem mdq_load_ptr_none {dtype : TileDType} {sh : TileShape}
    (nm : RegName) (t : BlockState) (pt : Tile .ptr sh)
    (hp : t.regs .ptr sh nm = some pt) :
    evalOp (Op.load dtype (MemAccess.ptr (Op.ref .ptr sh nm)) MaskOpt.none) t
      = some (⟨fun i => t.readMemValue dtype (pt.data i).1 (pt.data i).2⟩ :
          Tile dtype sh) := by
  simp only [evalOp, evalOp_ref, hp]
  rfl

/-- `.ptr` load with `mask=` / `other=`: masked-out lanes take the `other` tile. -/
private theorem mdq_load_ptr_maskOther {sh : TileShape}
    (nm : RegName) (t : BlockState) (pt : Tile .ptr sh)
    (mk : Op .bool sh) (ot : Op .real sh) (mv : Tile .bool sh) (ov : Tile .real sh)
    (hp : t.regs .ptr sh nm = some pt)
    (hm : evalOp mk t = some mv) (ho : evalOp ot t = some ov) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr sh nm))
        (MaskOpt.maskOther mk ot)) t
      = some (⟨fun i => if mv.data i then
            t.readMemValue .real (pt.data i).1 (pt.data i).2 else ov.data i⟩ :
          Tile .real sh) := by
  simp only [evalOp, evalOp_ref, hp, hm, ho]
  rfl

/-- Pointer advance / offset. -/
private theorem mdq_ptrAdd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (pnm : RegName) (t : BlockState) (pt : Tile .ptr a) (off : Op .nat b)
    (ov : Tile .nat b)
    (hp : t.regs .ptr a pnm = some pt) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ref .ptr a pnm) off) t = some (Tile.ptrAdd bc pt ov) := by
  simp only [evalOp, evalOp_ref, hp, ho]
  rfl

/-- `&` on the `nat` channel — there is no `evalOp_bitAnd` in the library. -/
private theorem mdq_bitAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.bitAnd bc x y) t = some (Tile.bop (· &&& ·) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `>>` on the `nat` channel. -/
private theorem mdq_shiftRight_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.shiftRight bc x y) t = some (Tile.bop (· >>> ·) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `Op.remap` — how the DSL rank-broadcasts the `[BM, 1]` load mask. -/
private theorem mdq_remap_eval {dtype : TileDType} {inS outS : TileShape}
    (map : TileIndex outS → TileIndex inS) (x : Op dtype inS) (t : BlockState)
    (vx : Tile dtype inS) (hx : evalOp x t = some vx) :
    evalOp (Op.remap outS map x) t = some (Tile.remap map vx) := by
  simp only [evalOp, hx]
  rfl

/-- `tl.broadcast_to`-style scalar fill, as the load's `other=0.0` produces. -/
private theorem mdq_broadcast_eval {dtype : TileDType} (e : Op dtype [])
    (sh : TileShape) (t : BlockState) (v : Tile dtype [])
    (hv : evalOp e t = some v) :
    evalOp (Op.broadcast e sh) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  simp only [evalOp, hv]
  rfl

/-! ## The scales / zeros channel

Both are `[BN]` tiles whose base pointer is set up once in the prologue and then
offset by `g_id * stride_*_g` inside the loop (or not offset at all, under
`NO_GROUPS`). Naming the two base address functions keeps the group row as the
only moving part. -/

/-- `scales_ptrs` lane `c`. -/
def mdqScalesAddr (stride_scales_n BN pn : Nat) (idx : TileIndex [BN]) : Nat :=
  (pn * BN + idx.1.val) * stride_scales_n

/-- `zeros_ptrs` lane `c` — eight zero-points share a word along N. -/
def mdqZerosAddr (stride_zeros_n BN pn : Nat) (idx : TileIndex [BN]) : Nat :=
  (pn * BN + idx.1.val) / 8 * stride_zeros_n

noncomputable def mdqScalesPtrs (scales : RegionName) (stride_scales_n BN pn : Nat) :
    Tile .ptr [BN] :=
  ⟨fun idx => (scales, mdqScalesAddr stride_scales_n BN pn idx)⟩

noncomputable def mdqZerosPtrs (zeros : Region .nat) (stride_zeros_n BN pn : Nat) :
    Tile .ptr [BN] :=
  ⟨fun idx => (Region.cast zeros, mdqZerosAddr stride_zeros_n BN pn idx)⟩

/-- The packed zero-point words as the load delivers them, before the nibble is
extracted — eight zero-points per word, so lanes `8c … 8c+7` share one. -/
noncomputable def mdqZerosRaw (s : BlockState) (zeros : Region .nat)
    (stride_zeros_g stride_zeros_n BN pn g : Nat) : Tile .nat [BN] :=
  ⟨fun idx => zerosWord s zeros stride_zeros_g stride_zeros_n BN pn g idx.1.val⟩

/-- `scales` as either branch leaves it, at group row `g`. -/
noncomputable def mdqScalesTile (s : BlockState) (scales : RegionName)
    (stride_scales_g stride_scales_n BN pn g : Nat) : Tile .real [BN] :=
  ⟨fun idx => some (scalesElem s scales stride_scales_g stride_scales_n BN pn g
      idx.1.val)⟩

/-- `zeros` as either branch leaves it — already **scaled**, since both branches
end with `zeros = zeros * scales`. -/
noncomputable def mdqZerosTile (s : BlockState) (scales : RegionName)
    (zeros : Region .nat)
    (stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN pn g : Nat) :
    Tile .real [BN] :=
  ⟨fun idx => some (zeroScaled s scales zeros stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BN pn g idx.1.val)⟩

/-- Offsetting `scales_ptrs` by `g * stride_scales_g` lands on `scalesElem`'s
address at group row `g`. -/
theorem mdqScalesPtrs_offset (scales : RegionName)
    (stride_scales_g stride_scales_n BN pn g : Nat) (idx : TileIndex [BN]) :
    (Tile.ptrAdd Broadcast.scalarR (mdqScalesPtrs scales stride_scales_n BN pn)
        (Tile.scalar (g * stride_scales_g))).data idx
      = (scales, (pn * BN + idx.1.val) * stride_scales_n + g * stride_scales_g) := by
  simp only [Tile.ptrAdd_data, mdqScalesPtrs, mdqScalesAddr, Tile.scalar,
    Broadcast.leftIndex]

/-- Likewise for `zeros_ptrs` and `zerosWord`'s address. -/
theorem mdqZerosPtrs_offset (zeros : Region .nat)
    (stride_zeros_g stride_zeros_n BN pn g : Nat) (idx : TileIndex [BN]) :
    (Tile.ptrAdd Broadcast.scalarR (mdqZerosPtrs zeros stride_zeros_n BN pn)
        (Tile.scalar (g * stride_zeros_g))).data idx
      = (Region.cast zeros,
          (pn * BN + idx.1.val) / 8 * stride_zeros_n + g * stride_zeros_g) := by
  simp only [Tile.ptrAdd_data, mdqZerosPtrs, mdqZerosAddr, Tile.scalar,
    Broadcast.leftIndex]

/-! ## The K-loop invariant

`i ≤ numPidK` is part of the predicate on purpose: `forRangeDyn_inv` concludes
only `stop ≤ final`, so carrying the upper bound is what pins `final = stop` and
lets the readout use `mdqAccTile_full`.

The `NO_GROUPS` case is the one that makes this invariant unlike a plain
carry-fold: under that flag `scales`/`zeros` are loop-invariant (loaded once,
before the loop, at group row `0`), while otherwise they are re-established by the
body at row `groupRow`. Both are covered by asking only that the registers hold
the *tiles for the row `groupRow NO_GROUPS groupsize BK i`* — which under
`NO_GROUPS` is `0` for every `i`. -/

/-- The state carried across K steps. -/
noncomputable def mdqInv (s0 : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M N K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK GM : Nat) (i : Nat) (s : BlockState) : Prop :=
  i ≤ numPidK K BK
  ∧ s.mem = s0.mem
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [] "bits" = some (Tile.scalar 4)
  ∧ s.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8)
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (pidM s0 M N BM BN GM))
  ∧ s.regs .nat [] "pid_n" = some (Tile.scalar (pidN s0 M N BM BN GM))
  ∧ s.regs .bool [BM, 1] "a_mask"
      = some (mdqAMask M BM (pidM s0 M N BM BN GM))
  ∧ s.regs .nat [BK] "shifter" = some (mdqShifter BK)
  ∧ s.regs .nat [BN] "zeros_shifter"
      = some (mdqZerosShifter BN (pidN s0 M N BM BN GM))
  ∧ s.regs .ptr [BN] "scales_ptrs"
      = some (mdqScalesPtrs scales stride_scales_n BN (pidN s0 M N BM BN GM))
  ∧ s.regs .ptr [BN] "zeros_ptrs"
      = some (mdqZerosPtrs zeros stride_zeros_n BN (pidN s0 M N BM BN GM))
  ∧ s.regs .ptr [BM, BK] "a_ptrs"
      = some (mdqAPtrs a stride_am stride_ak BM BK (pidM s0 M N BM BN GM) i)
  ∧ s.regs .ptr [BK, BN] "b_ptrs"
      = some (mdqBPtrs b stride_bk stride_bn BN BK (pidN s0 M N BM BN GM) i)
  ∧ s.regs .real [BM, BN] "accumulator"
      = some (mdqAccTile s0 a scales b zeros NO_GROUPS M groupsize stride_am
          stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
          stride_zeros_g stride_zeros_n BM BN BK (pidM s0 M N BM BN GM)
          (pidN s0 M N BM BN GM) i)
  ∧ (NO_GROUPS = Bool.true →
      s.regs .real [BN] "scales"
        = some (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN
            (pidN s0 M N BM BN GM) 0)
      ∧ s.regs .real [BN] "zeros"
        = some (mdqZerosTile s0 scales zeros stride_scales_g stride_scales_n
            stride_zeros_g stride_zeros_n BN (pidN s0 M N BM BN GM) 0))

/-- Under `NO_GROUPS` the group row is `0` at every step, so the loop-invariant
pre-load is exactly what every iteration needs. -/
theorem groupRow_of_noGroups (groupsize BK i : Nat) :
    groupRow Bool.true groupsize BK i = 0 := by
  simp [groupRow]

/-- `mdqInv` pins memory as a function, so every read agrees with the launch
state — including the `.nat` reads the packed channels use. -/
theorem mdqInv_readMemValue (s0 : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M N K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK GM i : Nat) (s : BlockState)
    (h : mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
      stride_zeros_n BM BN BK GM i s)
    (dtype : TileDType) (rg : RegionName) (off : Nat) :
    s.readMemValue dtype rg off = s0.readMemValue dtype rg off := by
  obtain ⟨_, hmem, _⟩ := h
  simp only [BlockState.readMemValue, BlockState.readMemAs, BlockState.readMemTyped,
    hmem]

/-! ### The two K-step loads, bridged to the named tiles -/

/-- The `a` load — pointer tile, rank-broadcast mask, `other=0.0` — lands exactly
on `mdqATile`, and on the *launch* state's memory. -/
private theorem mdq_aLoad_eq (s0 : BlockState) (a : RegionName) (t : BlockState)
    (M stride_am stride_ak BM BK pm i : Nat)
    (hmem : t.mem = s0.mem)
    (hap : t.regs .ptr [BM, BK] "a_ptrs"
      = some (mdqAPtrs a stride_am stride_ak BM BK pm i))
    (hmask : t.regs .bool [BM, 1] "a_mask" = some (mdqAMask M BM pm)) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (MaskOpt.maskOther
          (Op.remap [BM, BK]
            (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
            (Op.ref .bool [BM, 1] "a_mask"))
          (Op.broadcast (Op.const 0.0) [BM, BK]))) t
      = some (mdqATile s0 a M stride_am stride_ak BM BK pm i) := by
  rw [mdq_load_ptr_maskOther "a_ptrs" t
    (mdqAPtrs a stride_am stride_ak BM BK pm i) _ _
    (Tile.remap (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
      (mdqAMask M BM pm))
    (⟨fun _ => some (0.0 : ℝ)⟩ : Tile .real [BM, BK])
    hap
    (mdq_remap_eval _ _ t _ (by rw [evalOp_ref]; exact hmask))
    (mdq_broadcast_eval _ _ t _ (evalOp_const 0.0 t))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [mdqATile, mdqAPtrs, Tile.remap, mdqAMask, Broadcast.leftIndex,
    BlockState.readMemValue_real, BlockState.readMem, hmem, decide_eq_true_eq]
  split
  · rfl
  · norm_num

/-- The packed-weight load lands on `mdqBRaw`, on the launch state's memory. -/
private theorem mdq_bLoad_eq (s0 : BlockState) (b : Region .nat) (t : BlockState)
    (stride_bk stride_bn BN BK pn i : Nat)
    (hmem : t.mem = s0.mem)
    (hbp : t.regs .ptr [BK, BN] "b_ptrs"
      = some (mdqBPtrs b stride_bk stride_bn BN BK pn i)) :
    evalOp (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        MaskOpt.none) t
      = some (mdqBRaw s0 b stride_bk stride_bn BN BK pn i) := by
  rw [mdq_load_ptr_none "b_ptrs" t (mdqBPtrs b stride_bk stride_bn BN BK pn i) hbp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [mdqBRaw, mdqBPtrs, BlockState.readMemValue, BlockState.readMemTyped,
    hmem]

/-- A `[BN]` `.real` load through an offset `scales_ptrs` lands on `mdqScalesTile`
at that group row. -/
private theorem mdq_scalesLoad_eq (s0 : BlockState) (scales : RegionName)
    (t : BlockState) (stride_scales_g stride_scales_n BN pn g : Nat)
    (hmem : t.mem = s0.mem)
    (hp : t.regs .ptr [BN] "ptr"
      = some (Tile.ptrAdd Broadcast.scalarR
          (mdqScalesPtrs scales stride_scales_n BN pn)
          (Tile.scalar (g * stride_scales_g)))) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN] "ptr")) MaskOpt.none) t
      = some (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN pn g) := by
  rw [mdq_load_ptr_none "ptr" t _ hp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [mdqScalesTile, scalesElem, mdqScalesPtrs_offset,
    BlockState.readMemValue_real, BlockState.readMem, hmem]

/-- The packed zero-point load through an offset `zeros_ptrs` lands on
`zerosWord` at that group row. -/
private theorem mdq_zerosLoad_eq (s0 : BlockState) (zeros : Region .nat)
    (t : BlockState) (stride_zeros_g stride_zeros_n BN pn g : Nat)
    (hmem : t.mem = s0.mem)
    (hp : t.regs .ptr [BN] "ptr"
      = some (Tile.ptrAdd Broadcast.scalarR
          (mdqZerosPtrs zeros stride_zeros_n BN pn)
          (Tile.scalar (g * stride_zeros_g)))) :
    evalOp (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BN] "ptr")) MaskOpt.none) t
      = some (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn g) := by
  rw [mdq_load_ptr_none "ptr" t _ hp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [mdqZerosRaw, zerosWord, mdqZerosPtrs_offset, BlockState.readMemValue,
    BlockState.readMemTyped, hmem]

/-- `//` on `nat` scalars — there is no `evalOp_floorDiv` in the library. -/
private theorem mdq_floorDiv_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.floorDiv IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.floorDiv IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `name * c` on a `nat` scalar register. -/
private theorem mdq_mulRef_eval (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `Op.natToReal`. -/
private theorem mdq_natToReal_eval {sh : TileShape} (x : Op .nat sh)
    (t : BlockState) (vx : Tile .nat sh) (hx : evalOp x t = some vx) :
    evalOp (Op.natToReal x) t = some (Tile.natToReal vx) := by
  simp only [evalOp, hx]
  rfl

/-- The `g_id` value the branch computes at K step `i`, which is exactly
`groupRow` on the non-`NO_GROUPS` side. -/
theorem groupRow_of_groups (groupsize BK i : Nat) :
    groupRow Bool.false groupsize BK i = i / (groupsize / BK) := by
  simp [groupRow]

/-- **The branch's last two statements.** Extract the zero-point nibble, then
scale it — i.e. `mdqZerosTile` is exactly that composition over the loaded word. -/
private theorem mdqZerosTile_eq (s0 : BlockState) (scales : RegionName)
    (zeros : Region .nat)
    (stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN pn g : Nat) :
    mdqZerosTile s0 scales zeros stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BN pn g
      = Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
          (Tile.natToReal
            (Tile.bop (· &&& ·) Broadcast.scalarR
              (Tile.bop (· >>> ·) (Broadcast.consSame Broadcast.nil)
                (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn g)
                (mdqZerosShifter BN pn))
              (Tile.scalar 15)))
          (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN pn g) := by
  apply Tile.ext
  intro idx
  simp only [mdqZerosTile, zeroScaled, zerosNibble, mdqScalesTile, mdqZerosRaw,
    mdqZerosShifter, Tile.bop_data, Tile.natToReal, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, WithBot.realMul]
  rfl

/-- `//` on two `nat` **scalars**, in the shape the group row is computed in. -/
private theorem mdq_floorDivScalar_eval (x y : Op .nat []) (t : BlockState)
    (u v : Nat) (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u / v)) := by
  rw [mdq_floorDiv_eval Broadcast.nil x y t _ _ hx hy]
  rfl

/-- `*` on two tiles, given both operand values — the library's `evalOp_mul` is
stated as a `do` block, which is not a `rw` target once the operands are known. -/
private theorem mdq_mulTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul h bc x y) t = some (Tile.bop h.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- `Stmt.ifThen`'s step equation — `stepStmt` is defined by well-founded
recursion, so the definitional unfolding has to be named. -/
private theorem mdq_ifThen_step (cond : Op .bool []) (body : List Stmt)
    (t : BlockState) :
    stepStmt (Stmt.ifThen cond body) t
      = (evalOp cond t).bind
          (fun c => if c.data PUnit.unit then stepStmts body t else some t) := by
  unfold stepStmt
  cases evalOp cond t <;> rfl

/-- The branch guard `not NO_GROUPS` is a closed scalar. -/
private theorem mdq_notFlag_eval (flag : Bool) (t : BlockState) :
    evalOp (Op.boolNot (Op.constBool flag)) t = some (Tile.scalar (!flag)) := by
  simp only [evalOp]
  rfl

/-! ### The `NO_GROUPS` branch

The one place where the two configurations of this kernel genuinely diverge, and
the crux of the step proof. Both paths are made to land on the *same* statement —
`scales` and `zeros` hold the tiles for group row `groupRow NO_GROUPS groupsize BK
i` — so everything downstream is written once:

* `NO_GROUPS = true`: the branch is skipped, the state is untouched, and the
  registers still hold the prologue's row-`0` pre-load, which `groupRow_of_noGroups`
  identifies with the row asked for.
* `NO_GROUPS = false`: the seven statements re-establish both registers at row
  `i / (groupsize / BK)`, which `groupRow_of_groups` identifies with the same.

The register-preservation clause is stated as a side condition on *names* rather
than as a list of the individual registers the rest of the body needs — the branch
writes only `g_id`, `ptr`, `scales`, `zeros`, and nothing else has to be enumerated. -/

/-! ### The dequantized weight tile and the accumulator step

The last three compute statements of the body. `b` is rewritten twice — once to
extract the nibble on the `nat` channel, once to scale and shift it onto the `real`
channel — and only then does `tl.dot` fire, so the dequantization is entirely
inside the K step and `bDequant` is the right unit to name. -/

/-- `b` after both dequant statements at K step `k`: the unpacked nibble scaled by
`scales`, minus the already-scaled `zeros`, both read at that step's group row. -/
noncomputable def mdqBDequantTile (s : BlockState) (scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (groupsize stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BN BK pn k : Nat) : Tile .real [BK, BN] :=
  ⟨fun idx => some (bDequant s scales b zeros NO_GROUPS groupsize stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN BK pn
      k idx.1.val idx.2.1.val)⟩

/-- The dequantized tile **is** the compiled `b * scales[None, :] - zeros[None, :]`,
with the group row supplied by `groupRow`. Both `[BN]` operands are rank-broadcast
along the K axis, which is why one group row serves the whole tile. -/
theorem mdqBDequantTile_eq (s : BlockState) (scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (groupsize stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BN BK pn k : Nat) :
    mdqBDequantTile s scales b zeros NO_GROUPS groupsize stride_bk stride_bn
        stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN BK pn k
      = Tile.bop NumericDType.real.sub
          (Broadcast.consR (Broadcast.consSame Broadcast.nil))
          (Tile.bop NumericDType.real.mul
            (Broadcast.consR (Broadcast.consSame Broadcast.nil))
            (Tile.natToReal (mdqBNibbleTile s b stride_bk stride_bn BN BK pn k))
            (Tile.expandDim ⟨0, by simp⟩
              (mdqScalesTile s scales stride_scales_g stride_scales_n BN pn
                (groupRow NO_GROUPS groupsize BK k))))
          (Tile.expandDim ⟨0, by simp⟩
            (mdqZerosTile s scales zeros stride_scales_g stride_scales_n
              stride_zeros_g stride_zeros_n BN pn
              (groupRow NO_GROUPS groupsize BK k))) := by
  apply Tile.ext
  intro idx
  simp only [mdqBDequantTile, bDequant, mdqBNibbleTile, mdqScalesTile, mdqZerosTile,
    Tile.bop_data, Tile.natToReal, Tile.expandDim_data, TileShape.dropInsertedIndex,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub, NumericDType.mul,
    WithBot.realSub, WithBot.realMul]
  rfl

/-- A `WithBot ℝ` sum of pointwise products of `some`s is the `some` of the `ℝ`
sum — the collapse `Tile.dot` needs, since every operand lane here is a loaded
(non-`⊥`) value. -/
private theorem mdq_coe_sum_mul {n : Nat} (f g : Fin n → ℝ) :
    (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
        Option.map₂ (fun x y : ℝ => x * y) (some (f e)) (some (g e)))
      = some (∑ e : Fin n, f e * g e) := by
  rw [show (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
        Option.map₂ (fun x y : ℝ => x * y) (some (f e)) (some (g e)))
      = (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
          (some (f e * g e) : WithBot ℝ)) from Finset.sum_congr rfl fun e _ => by simp]
  show (Finset.univ.sum fun e => ((f e * g e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]
  rfl

/-- **The accumulator statement.** `accumulator += tl.dot(a, b)` extends the partial
sum by exactly one `accStep`. -/
theorem mdqAccTile_dot_succ (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn i : Nat) :
    Tile.bop NumericDType.real.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (mdqAccTile s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
          stride_bk stride_bn stride_scales_g stride_scales_n
          stride_zeros_g stride_zeros_n BM BN BK pm pn i)
        (Tile.dot [] (mdqATile s a M stride_am stride_ak BM BK pm i)
          (mdqBDequantTile s scales b zeros NO_GROUPS groupsize stride_bk stride_bn
            stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN BK pn i))
      = mdqAccTile s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
          stride_bk stride_bn stride_scales_g stride_scales_n
          stride_zeros_g stride_zeros_n BM BN BK pm pn (i + 1) := by
  apply Tile.ext
  intro idx
  obtain ⟨r, c, u⟩ := idx
  rw [mdqAccTile_succ]
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, mdqAccTile,
    NumericDType.add, WithBot.realAdd]
  -- `erw`: `Tile.dot`'s operand shapes are `[] ++ [M, K]`, so `Tile.dot_nil_data`
  -- does not fire under `rw` / `simp only`.
  erw [Tile.dot_nil_data]
  simp only [mdqATile_data, mdqBDequantTile]
  rw [mdq_coe_sum_mul]
  simp [accStep]

/-! ### The remaining eval recipes -/

/-- `-` on two tiles, both operand values known. -/
private theorem mdq_subTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.sub h bc x y) t = some (Tile.bop h.sub bc vx vy) := by
  rw [evalOp_sub, hx, hy]
  rfl

/-- `+` on two tiles, both operand values known. -/
private theorem mdq_addTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add h bc x y) t = some (Tile.bop h.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

/-- `*` on two `nat` **scalars** — the shape both pointer advances use. -/
private theorem mdq_mulScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mul .nat Broadcast.nil x y) t = some (Tile.scalar (u * v)) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- `[:, None]` / `[None, :]`.  The axis binder is spelled `ax`: the DSL claims
`axis` as a keyword-argument token, so it cannot be used as an identifier here. -/
private theorem mdq_expandDim_eval {dtype : TileDType} {sh : TileShape}
    (ax : Fin (sh.length + 1)) (x : Op dtype sh) (t : BlockState)
    (v : Tile dtype sh) (hv : evalOp x t = some v) :
    evalOp (Op.expandDim ax x) t = some (Tile.expandDim ax v) := by
  rw [evalOp_expandDim, hv]
  rfl

/-- `tl.dot` at rank 2. `erw`, not `rw`: the operand shapes are `[] ++ [M, K]`,
which does not unfold at reducible transparency, so `evalOp_dot` silently fails to
fire under `rw` / `simp only`. -/
private theorem mdq_dot_eval {M K N : Nat} (x : Op .real [M, K]) (y : Op .real [K, N])
    (t : BlockState) (vx : Tile .real [M, K]) (vy : Tile .real [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dot (batch := []) x y) t = some (Tile.dot [] vx vy) := by
  erw [evalOp_dot, hx, hy]
  rfl

/-! ### The K step

One iteration takes `mdqInv` at `i` to `mdqInv` at `i + 1`. The eight statements are
walked in order; the only branching is the `NO_GROUPS` reload, and
`mdq_groupBranch_run` has already collapsed both of its paths to one conclusion, so
statements 4–8 are walked once. -/

/-- The loop combinator writes the induction variable before each iteration, and
`mdqInv` constrains no register named `"k"`, so the invariant survives that write. -/
theorem mdqInv_setReg_k (s0 : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M N K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK GM i j : Nat) (s : BlockState)
    (h : mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
      stride_zeros_n BM BN BK GM i s) :
    mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
      stride_zeros_n BM BN BK GM i (s.setReg "k" .nat [] (Tile.scalar j)) := by
  obtain ⟨hle, hmem, hpids, hbits, hipb, hpm, hpn, hmask, hshift, hzshift,
    hspt, hzpt, hapt, hbpt, hacc, hpreload⟩ := h
  exact ⟨hle, hmem, hpids, by simpa using hbits, by simpa using hipb,
    by simpa using hpm, by simpa using hpn, by simpa using hmask,
    by simpa using hshift, by simpa using hzshift, by simpa using hspt,
    by simpa using hzpt, by simpa using hapt, by simpa using hbpt,
    by simpa using hacc, fun hf => by simpa using hpreload hf⟩

/-! ### Collapsing the K loop

`forRangeDyn_inv` concludes only `stop ≤ final`, so the readout needs the other
inequality; that is what the invariant's own `i ≤ numPidK K BK` conjunct is for —
together they pin `final = numPidK K BK` and hand the accumulator to
`mdqAccTile_full`. -/

/-! ## The prologue's building blocks

Every prologue statement is one of two kinds: a `nat` **scalar** step (the pid
swizzle and the three `tl.cdiv`s), or a shaped **index tile** (the offset vectors,
the four pointer tiles, the mask and the two shifters). The scalar family all has
the same shape — `Tile.bop f Broadcast.nil (Tile.scalar u) (Tile.scalar v)` is
`Tile.scalar (f u v)` by `rfl` — so those recipes are listed together, followed by
one `evalOp` lemma per index tile that lands directly on the named tile. -/

/-- `setReg` leaves memory alone, at **function** level. The library's
`setReg_mem` is pointwise, and closing a twelve-deep tower's `t.mem = s.mem` by a
single `rfl` overruns `whnf`; twelve cheap rewrites do not. -/
private theorem mdq_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-- `Op.div` on any numeric channel (this is what `tl.cdiv` expands to; `//` is
`Op.floorDiv`). -/
private theorem mdq_divTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.div h bc x y) t = some (Tile.bop h.div bc vx vy) := by
  rw [evalOp_div, hx, hy]
  rfl

/-- `%` on the `nat` channel. -/
private theorem mdq_mod_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mod IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.mod IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `<`. -/
private theorem mdq_ltTile_eval {dtype : TileDType} (h : ComparableDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.lt h bc x y) t = some (Tile.cop h.lt bc vx vy) := by
  rw [evalOp_lt, hx, hy]
  rfl

/-- `tl.where` — how `min(a, b)` is lowered, there being no `Op.min`. -/
private theorem mdq_where_eval {dtype : TileDType} {sh : TileShape}
    (c : Op .bool sh) (x y : Op dtype sh) (t : BlockState)
    (vc : Tile .bool sh) (vx vy : Tile dtype sh)
    (hc : evalOp c t = some vc) (hx : evalOp x t = some vx)
    (hy : evalOp y t = some vy) :
    evalOp (Op.where c x y) t = some (Tile.select vc vx vy) := by
  rw [evalOp_where, hc, hx, hy]
  rfl

/-- `tl.zeros` / `tl.full`. -/
private theorem mdq_full_eval {dtype : TileDType} (sh : TileShape) (e : Op dtype [])
    (t : BlockState) (v : Tile dtype []) (hv : evalOp e t = some v) :
    evalOp (Op.full sh e) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  rw [evalOp_full, hv]
  rfl

/-- A pointer tile built from a bare region base, as every `R + offs` in the
prologue is. -/
private theorem mdq_ptrAddBase_eval {d : TileDType} {b out : TileShape}
    (bc : Broadcast [] b out) (rg : Region d) (t : BlockState)
    (off : Op .nat b) (ov : Tile .nat b) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ptrBase rg) off) t
      = some (Tile.ptrAdd bc (Tile.scalar ((Region.cast rg : RegionName), 0)) ov) := by
  simp only [evalOp, ho]
  rfl

/-! ### `nat` scalars -/

private theorem mdq_addScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.add .nat Broadcast.nil x y) t = some (Tile.scalar (u + v)) := by
  rw [mdq_addTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mdq_subScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.sub .nat Broadcast.nil x y) t = some (Tile.scalar (u - v)) := by
  rw [mdq_subTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mdq_divScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.div .nat Broadcast.nil x y) t = some (Tile.scalar (u / v)) := by
  rw [mdq_divTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mdq_modScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u % v)) := by
  rw [mdq_mod_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mdq_ltScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.lt ComparableDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (decide (u < v))) := by
  rw [mdq_ltTile_eval ComparableDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mdq_whereScalarNat_eval (c : Op .bool []) (x y : Op .nat [])
    (t : BlockState) (cv : Bool) (u v : Nat)
    (hc : evalOp c t = some (Tile.scalar cv))
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.where c x y) t = some (Tile.scalar (if cv then u else v)) := by
  rw [mdq_where_eval c x y t _ _ _ hc hx hy]
  rfl

/-- `min` is spelled as a `tl.where` by the DSL; this is the arithmetic identity
that turns the lowered form back into `groupSizeM`'s `min`. -/
private theorem mdq_min_as_where (u v : Nat) : (if u < v then u else v) = min u v := by
  rcases Nat.lt_or_ge u v with h | h
  · rw [if_pos h]; omega
  · rw [if_neg (by omega)]; omega

/-! ### Index tiles -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)`, and `tl.arange` on its own at `base = 0`. -/
def mdqOffsTile (base BD : Nat) : Tile .nat [BD] := ⟨fun idx => base + idx.1.val⟩

private theorem mdq_arange_eq_offs (BD : Nat) :
    (Tile.vec (fun i => (i.val : Nat)) : Tile .nat [BD]) = mdqOffsTile 0 BD := by
  apply Tile.ext
  intro idx
  simp [mdqOffsTile, Tile.vec]

private theorem mdq_offs_eval (nm : RegName) (t : BlockState) (BD base c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c))
        (Op.arange BD)) t
      = some (mdqOffsTile (base * c) BD) := by
  rw [mdq_addTile_eval NumericDType.nat Broadcast.scalarL _ _ t
    (Tile.scalar (base * c)) (Tile.vec (fun i => (i.val : Nat)))
    (mdq_mulScalarNat_eval _ _ t base c (by rw [evalOp_ref]; exact hr)
      (evalOp_constNat _ _))
    (evalOp_arange _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqOffsTile, Tile.vec, Broadcast.rightIndex, NumericDType.add]

/-- `a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak`. -/
private theorem mdq_aPtrsInit_eval (a : RegionName) (t : BlockState)
    (stride_am stride_ak BM BK pm : Nat)
    (ham : t.regs .nat [BM] "offs_am" = some (mdqOffsTile (pm * BM) BM))
    (hok : t.regs .nat [BK] "offs_k" = some (mdqOffsTile 0 BK)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase a)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat stride_am))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_ak)))) t
      = some (mdqAPtrs a stride_am stride_ak BM BK pm 0) := by
  rw [mdq_ptrAddBase_eval _ _ t _ _
    (mdq_addTile_eval NumericDType.nat _ _ _ t _ _
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar stride_am)
        (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact ham))
        (evalOp_constNat _ _))
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar stride_ak)
        (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hok))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqAPtrs, mdqAAddr, mdqOffsTile, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `a_mask = offs_am[:, None] < M`. -/
private theorem mdq_aMaskInit_eval (t : BlockState) (M BM pm : Nat)
    (ham : t.regs .nat [BM] "offs_am" = some (mdqOffsTile (pm * BM) BM)) :
    evalOp ((Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
        (Op.constNat M) : Op .bool [BM, 1])) t
      = some (mdqAMask M BM pm) := by
  rw [mdq_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar M)
    (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact ham))
    (evalOp_constNat _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqAMask, mdqOffsTile, Tile.cop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, ComparableDType.lt]

/-- `b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn`. -/
private theorem mdq_bPtrsInit_eval (b : Region .nat) (t : BlockState)
    (stride_bk stride_bn BN BK pn : Nat)
    (hok : t.regs .nat [BK] "offs_k" = some (mdqOffsTile 0 BK))
    (hbn : t.regs .nat [BN] "offs_bn" = some (mdqOffsTile (pn * BN) BN))
    (hipb : t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase b)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
              (Op.ref .nat [] "infearure_per_bits"))
            (Op.constNat stride_bk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat stride_bn)))) t
      = some (mdqBPtrs b stride_bk stride_bn BN BK pn 0) := by
  rw [mdq_ptrAddBase_eval _ _ t _ _
    (mdq_addTile_eval NumericDType.nat _ _ _ t _ _
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar stride_bk)
        (mdq_floorDiv_eval Broadcast.scalarR _ _ t _ (Tile.scalar 8)
          (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hok))
          (by rw [evalOp_ref]; exact hipb))
        (evalOp_constNat _ _))
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar stride_bn)
        (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hbn))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqBPtrs, mdqBAddr, mdqOffsTile, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `scales_ptrs = scales_ptr + offs_bn * stride_scales_n`. -/
private theorem mdq_scalesPtrsInit_eval (scales : RegionName) (t : BlockState)
    (stride_scales_n BN pn : Nat)
    (hbn : t.regs .nat [BN] "offs_bn" = some (mdqOffsTile (pn * BN) BN)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase scales)
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
          (Op.constNat stride_scales_n))) t
      = some (mdqScalesPtrs scales stride_scales_n BN pn) := by
  rw [mdq_ptrAddBase_eval _ _ t _ _
    (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar stride_scales_n) (by rw [evalOp_ref]; exact hbn)
      (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqScalesPtrs, mdqScalesAddr, mdqOffsTile, Tile.ptrAdd_data, Tile.bop_data,
    Broadcast.leftIndex, NumericDType.mul]

/-- `zeros_ptrs = zeros_ptr + (offs_bn // 8) * stride_zeros_n`. -/
private theorem mdq_zerosPtrsInit_eval (zeros : Region .nat) (t : BlockState)
    (stride_zeros_n BN pn : Nat)
    (hbn : t.regs .nat [BN] "offs_bn" = some (mdqOffsTile (pn * BN) BN))
    (hipb : t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase zeros)
        (Op.mul .nat Broadcast.scalarR
          (Op.floorDiv IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
            (Op.ref .nat [] "infearure_per_bits"))
          (Op.constNat stride_zeros_n))) t
      = some (mdqZerosPtrs zeros stride_zeros_n BN pn) := by
  rw [mdq_ptrAddBase_eval _ _ t _ _
    (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar stride_zeros_n)
      (mdq_floorDiv_eval Broadcast.scalarR _ _ t _ (Tile.scalar 8)
        (by rw [evalOp_ref]; exact hbn) (by rw [evalOp_ref]; exact hipb))
      (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqZerosPtrs, mdqZerosAddr, mdqOffsTile, Tile.ptrAdd_data, Tile.bop_data,
    Broadcast.leftIndex, NumericDType.mul]

/-- `shifter = (offs_k % infearure_per_bits) * bits`. -/
private theorem mdq_shifterInit_eval (t : BlockState) (BK : Nat)
    (hok : t.regs .nat [BK] "offs_k" = some (mdqOffsTile 0 BK))
    (hipb : t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8))
    (hbits : t.regs .nat [] "bits" = some (Tile.scalar 4)) :
    evalOp (Op.mul .nat Broadcast.scalarR
        (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BK] "offs_k")
          (Op.ref .nat [] "infearure_per_bits"))
        (Op.ref .nat [] "bits")) t
      = some (mdqShifter BK) := by
  rw [mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar 4)
    (mdq_mod_eval Broadcast.scalarR _ _ t _ (Tile.scalar 8)
      (by rw [evalOp_ref]; exact hok) (by rw [evalOp_ref]; exact hipb))
    (by rw [evalOp_ref]; exact hbits)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqShifter, mdqOffsTile, Tile.bop_data, Broadcast.leftIndex,
    NumericDType.mul]

/-- `zeros_shifter = (offs_bn % infearure_per_bits) * bits`. -/
private theorem mdq_zerosShifterInit_eval (t : BlockState) (BN pn : Nat)
    (hbn : t.regs .nat [BN] "offs_bn" = some (mdqOffsTile (pn * BN) BN))
    (hipb : t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8))
    (hbits : t.regs .nat [] "bits" = some (Tile.scalar 4)) :
    evalOp (Op.mul .nat Broadcast.scalarR
        (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
          (Op.ref .nat [] "infearure_per_bits"))
        (Op.ref .nat [] "bits")) t
      = some (mdqZerosShifter BN pn) := by
  rw [mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar 4)
    (mdq_mod_eval Broadcast.scalarR _ _ t _ (Tile.scalar 8)
      (by rw [evalOp_ref]; exact hbn) (by rw [evalOp_ref]; exact hipb))
    (by rw [evalOp_ref]; exact hbits)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqZerosShifter, mdqOffsTile, Tile.bop_data, Broadcast.leftIndex,
    NumericDType.mul]

/-! ### The prologue's `NO_GROUPS` pre-load

The mirror of `mdq_groupBranch_run`, on the positive guard and at group row `0`. Its
conclusion is deliberately weaker: nothing after it *needs* `scales` / `zeros` unless
`NO_GROUPS` holds, because otherwise the K loop reloads them every step. -/

/-- A `[BN]` `.real` load straight off `scales_ptrs` — group row `0`. -/
private theorem mdq_scalesLoad0_eq (s0 : BlockState) (scales : RegionName)
    (t : BlockState) (stride_scales_g stride_scales_n BN pn : Nat)
    (hmem : t.mem = s0.mem)
    (hp : t.regs .ptr [BN] "scales_ptrs"
      = some (mdqScalesPtrs scales stride_scales_n BN pn)) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN] "scales_ptrs"))
        MaskOpt.none) t
      = some (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN pn 0) := by
  rw [mdq_load_ptr_none "scales_ptrs" t _ hp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [mdqScalesTile, scalesElem, mdqScalesPtrs, mdqScalesAddr,
    BlockState.readMemValue_real, BlockState.readMem, hmem, Nat.zero_mul, Nat.add_zero]

/-- The packed zero-point load straight off `zeros_ptrs` — group row `0`. -/
private theorem mdq_zerosLoad0_eq (s0 : BlockState) (zeros : Region .nat)
    (t : BlockState) (stride_zeros_g stride_zeros_n BN pn : Nat)
    (hmem : t.mem = s0.mem)
    (hp : t.regs .ptr [BN] "zeros_ptrs"
      = some (mdqZerosPtrs zeros stride_zeros_n BN pn)) :
    evalOp (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BN] "zeros_ptrs"))
        MaskOpt.none) t
      = some (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn 0) := by
  rw [mdq_load_ptr_none "zeros_ptrs" t _ hp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [mdqZerosRaw, zerosWord, mdqZerosPtrs, mdqZerosAddr,
    BlockState.readMemValue, BlockState.readMemTyped, hmem, Nat.zero_mul, Nat.add_zero]

/-- `Op.constBool`. -/
private theorem mdq_constBool_eval (flag : Bool) (t : BlockState) :
    evalOp (Op.constBool flag) t = some (Tile.scalar flag) := by
  simp only [evalOp]

/-! ### The pid swizzle, statement by statement

Each recipe lands on the **named** quantity rather than on raw arithmetic, so the
walk below carries `numPidM` / `groupId` / `pidM` and never a `(M + BM - 1) / BM`.
Each is definitionally the arithmetic it names, which is why the `rw` closes. -/

private theorem mdq_cdiv_eval (t : BlockState) (X BX : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat X) (Op.constNat BX)) (Op.constNat 1))
        (Op.constNat BX)) t
      = some (Tile.scalar ((X + BX - 1) / BX)) :=
  mdq_divScalarNat_eval _ _ t (X + BX - 1) BX
    (mdq_subScalarNat_eval _ _ t (X + BX) 1
      (mdq_addScalarNat_eval _ _ t X BX (evalOp_constNat _ _) (evalOp_constNat _ _))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)

private theorem mdq_numPidM_eval (t : BlockState) (M BM : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
        (Op.constNat BM)) t
      = some (Tile.scalar (numPidM M BM)) := mdq_cdiv_eval t M BM

private theorem mdq_numPidN_eval (t : BlockState) (N BN : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
        (Op.constNat BN)) t
      = some (Tile.scalar (numPidN N BN)) := mdq_cdiv_eval t N BN

private theorem mdq_numPidK_eval (t : BlockState) (K BK : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
        (Op.constNat BK)) t
      = some (Tile.scalar (numPidK K BK)) := mdq_cdiv_eval t K BK

private theorem mdq_numPidInGroup_eval (t : BlockState) (N BN GM : Nat)
    (hnpn : t.regs .nat [] "num_pid_n" = some (Tile.scalar (numPidN N BN))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat GM)
        (Op.ref .nat [] "num_pid_n")) t
      = some (Tile.scalar (numPidInGroup N BN GM)) := by
  rw [mdq_mulScalarNat_eval _ _ t GM (numPidN N BN) (evalOp_constNat _ _)
    (by rw [evalOp_ref]; exact hnpn)]
  rfl

private theorem mdq_groupId_eval (s t : BlockState) (N BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hnig : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (numPidInGroup N BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "num_pid_in_group")) t
      = some (Tile.scalar (groupId s N BN GM)) := by
  rw [mdq_floorDivScalar_eval _ _ t (s.pids 0) (numPidInGroup N BN GM)
    (by rw [evalOp_ref]; exact hpid) (by rw [evalOp_ref]; exact hnig)]
  rfl

private theorem mdq_firstPidM_eval (s t : BlockState) (N BN GM : Nat)
    (hgid : t.regs .nat [] "group_id" = some (Tile.scalar (groupId s N BN GM))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
        (Op.constNat GM)) t
      = some (Tile.scalar (firstPidM s N BN GM)) := by
  rw [mdq_mulScalarNat_eval _ _ t (groupId s N BN GM) GM
    (by rw [evalOp_ref]; exact hgid) (evalOp_constNat _ _)]
  rfl

private theorem mdq_groupSizeM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hnpm : t.regs .nat [] "num_pid_m" = some (Tile.scalar (numPidM M BM)))
    (hfpm : t.regs .nat [] "first_pid_m"
      = some (Tile.scalar (firstPidM s N BN GM))) :
    evalOp (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
            (Op.ref .nat [] "first_pid_m"))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
          (Op.ref .nat [] "first_pid_m"))
        (Op.constNat GM)) t
      = some (Tile.scalar (groupSizeM s M N BM BN GM)) := by
  have hsub : evalOp (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
      (Op.ref .nat [] "first_pid_m")) t
      = some (Tile.scalar (numPidM M BM - firstPidM s N BN GM)) :=
    mdq_subScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hnpm)
      (by rw [evalOp_ref]; exact hfpm)
  rw [mdq_whereScalarNat_eval _ _ _ t
    (decide (numPidM M BM - firstPidM s N BN GM < GM))
    (numPidM M BM - firstPidM s N BN GM) GM
    (mdq_ltScalarNat_eval _ _ t _ _ hsub (evalOp_constNat _ _)) hsub
    (evalOp_constNat _ _)]
  simp only [decide_eq_true_eq, mdq_min_as_where, groupSizeM]

private theorem mdq_pidM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hfpm : t.regs .nat [] "first_pid_m" = some (Tile.scalar (firstPidM s N BN GM)))
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hgsm : t.regs .nat [] "group_size_m"
      = some (Tile.scalar (groupSizeM s M N BM BN GM))) :
    evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "group_size_m"))) t
      = some (Tile.scalar (pidM s M N BM BN GM)) := by
  rw [mdq_addScalarNat_eval _ _ t (firstPidM s N BN GM)
    (s.pids 0 % groupSizeM s M N BM BN GM) (by rw [evalOp_ref]; exact hfpm)
    (mdq_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hgsm))]
  rfl

private theorem mdq_pidN_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hnig : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (numPidInGroup N BN GM)))
    (hgsm : t.regs .nat [] "group_size_m"
      = some (Tile.scalar (groupSizeM s M N BM BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "group_size_m")) t
      = some (Tile.scalar (pidN s M N BM BN GM)) := by
  rw [mdq_floorDivScalar_eval _ _ t (s.pids 0 % numPidInGroup N BN GM)
    (groupSizeM s M N BM BN GM)
    (mdq_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hnig))
    (by rw [evalOp_ref]; exact hgsm)]
  rfl

private theorem mdq_offsK_eval (t : BlockState) (BK : Nat) :
    evalOp (Op.arange BK) t = some (mdqOffsTile 0 BK) := by
  rw [evalOp_arange, mdq_arange_eq_offs]

/-! ### The prologue walks -/

/-! ## The output store

A masked `.ptr` store, so — unlike `chunk_bwd_dqkg`'s block-pointer stores — the
`c_mask` is the only gate and there is no boundary check to discharge. The lane-to-
address map has to be injective for the readback to name a unique lane; that is the
headline's `hInj`, and `cAddr_injective` discharges it from the two conditions a
row-major `C` satisfies. -/

/-- The `C` pointer tile. -/
noncomputable def mdqCPtrs (c : RegionName) (stride_cm stride_cn BM BN pm pn : Nat) :
    Tile .ptr [BM, BN] :=
  ⟨fun idx => (c, cAddr stride_cm stride_cn BM BN pm pn idx)⟩

/-- `c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)`. -/
def mdqCMask (M N BM BN pm pn : Nat) : Tile .bool [BM, BN] :=
  ⟨fun idx => decide (pm * BM + idx.1.val < M) && decide (pn * BN + idx.2.1.val < N)⟩

/-- The post-store state: one masked scatter over the `[BM, BN]` output tile. -/
noncomputable def mdqStoreState (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat)
    (f : TileIndex [BM, BN] → ℝ) (t : BlockState) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl
    (fun acc i => if pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N then
        acc.writeMem c (cAddr stride_cm stride_cn BM BN pm pn i) (f i)
      else acc) t

/-- `&` on the bool channel. -/
private theorem mdq_boolAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .bool a) (y : Op .bool b) (t : BlockState)
    (vx : Tile .bool a) (vy : Tile .bool b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.boolAnd bc x y) t
      = some (Tile.bop (fun u v : Bool => u && v) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

private theorem mdq_cPtrsInit_eval (c : RegionName) (t : BlockState)
    (stride_cm stride_cn BM BN pm pn : Nat)
    (hcm : t.regs .nat [BM] "offs_cm" = some (mdqOffsTile (pm * BM) BM))
    (hcn : t.regs .nat [BN] "offs_cn" = some (mdqOffsTile (pn * BN) BN)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase c)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cm)
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cn)
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) t
      = some (mdqCPtrs c stride_cm stride_cn BM BN pm pn) := by
  rw [mdq_ptrAddBase_eval _ _ t _ _
    (mdq_addTile_eval NumericDType.nat _ _ _ t _ _
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
        (Tile.scalar stride_cm) _ (evalOp_constNat _ _)
        (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcm)))
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
        (Tile.scalar stride_cn) _ (evalOp_constNat _ _)
        (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcn))))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqCPtrs, cAddr, mdqOffsTile, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

private theorem mdq_cMaskInit_eval (t : BlockState) (M N BM BN pm pn : Nat)
    (hcm : t.regs .nat [BM] "offs_cm" = some (mdqOffsTile (pm * BM) BM))
    (hcn : t.regs .nat [BN] "offs_cn" = some (mdqOffsTile (pn * BN) BN)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        ((Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm"))
          (Op.constNat M) : Op .bool [BM, 1]))
        ((Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))
          (Op.constNat N) : Op .bool [1, BN]))) t
      = some (mdqCMask M N BM BN pm pn) := by
  rw [mdq_boolAnd_eval _ _ _ t _ _
    (mdq_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar M)
      (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcm))
      (evalOp_constNat _ _))
    (mdq_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar N)
      (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcn))
      (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqCMask, mdqOffsTile, Tile.bop_data, Tile.cop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

private theorem mdq_store_eq (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (vt : Tile .real [BM, BN]) (f : TileIndex [BM, BN] → ℝ)
    (hfv : ∀ i, vt.data i = some (f i))
    (hcp : t.regs .ptr [BM, BN] "c_ptrs"
      = some (mdqCPtrs c stride_cm stride_cn BM BN pm pn))
    (hcmask : t.regs .bool [BM, BN] "c_mask" = some (mdqCMask M N BM BN pm pn))
    (hv : t.regs .real [BM, BN] "accumulator" = some vt) :
    stepStmt (Stmt.store .real [BM, BN]
        (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .real [BM, BN] "accumulator")
        (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask"))) t
      = some (mdqStoreState c M N stride_cm stride_cn BM BN pm pn f t) := by
  unfold stepStmt mdqStoreState
  simp only [evalOp_ref, hv, hcp, hcmask, Option.map_some]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BN])) ?_)
  funext acc i
  obtain ⟨r, cc, u⟩ := i
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · simp only [mdqCMask, if_pos hb, mdqCPtrs, BlockState.writeMemTyped_real, hfv]
    simp [hb.1, hb.2]
  · simp only [mdqCMask, if_neg hb, mdqCPtrs]
    rw [if_neg]
    intro hcon
    exact hb (by simpa using hcon)

private theorem mdq_store_props (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (f : TileIndex [BM, BN] → ℝ)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn i)) :
    ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) →
      (mdqStoreState c M N stride_cm stride_cn BM BN pm pn f t).readMem c
          (cAddr stride_cm stride_cn BM BN pm pn i) = f i := by
  classical
  intro i hi
  unfold mdqStoreState
  have h := BlockState.scatter_readback_prop_masked_nd (region := c) t
    (fun j : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn j) f
    (fun j : TileIndex [BM, BN] => pm * BM + j.1.val < M ∧ pn * BN + j.2.1.val < N)
    hInj i
  rw [h, if_pos hi]

/-- `hInj` for a row-major `C`: distinct output lanes get distinct addresses as soon
as the column stride is positive and one block row fits inside the row stride. -/
theorem cAddr_injective (stride_cm stride_cn BM BN pm pn : Nat)
    (hcn : 0 < stride_cn) (hfit : BN * stride_cn ≤ stride_cm) :
    Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn i) := by
  intro i j hij
  obtain ⟨r₁, c₁, u₁⟩ := i
  obtain ⟨r₂, c₂, u₂⟩ := j
  simp only [cAddr] at hij
  -- Every product below is opaque to `omega`, so name them first.
  have hexp : ∀ r c : Nat, stride_cm * (pm * BM + r) + stride_cn * (pn * BN + c)
      = stride_cm * (pm * BM) + stride_cm * r
        + (stride_cn * (pn * BN) + stride_cn * c) := by
    intro r c
    rw [Nat.mul_add, Nat.mul_add]
  rw [hexp, hexp] at hij
  have key : stride_cm * r₁.val + stride_cn * c₁.val
      = stride_cm * r₂.val + stride_cn * c₂.val := by omega
  -- A whole block row fits under one row stride, so the row index is forced first.
  have hrow : ∀ c : Fin BN, stride_cn * c.val < stride_cm := fun c =>
    lt_of_lt_of_le
      (by rw [Nat.mul_comm]; exact Nat.mul_lt_mul_of_lt_of_le c.isLt (le_refl _) hcn)
      hfit
  have hr : r₁.val = r₂.val := by
    rcases Nat.lt_trichotomy r₁.val r₂.val with h | h | h
    · have : stride_cm * r₁.val + stride_cm ≤ stride_cm * r₂.val := by
        rw [← Nat.mul_succ]
        exact Nat.mul_le_mul_left _ h
      have := hrow c₁
      omega
    · exact h
    · have : stride_cm * r₂.val + stride_cm ≤ stride_cm * r₁.val := by
        rw [← Nat.mul_succ]
        exact Nat.mul_le_mul_left _ h
      have := hrow c₂
      omega
  have hc : c₁.val = c₂.val := by
    have : stride_cn * c₁.val = stride_cn * c₂.val := by rw [hr] at key; omega
    exact Nat.eq_of_mul_eq_mul_left hcn this
  simp only [Prod.mk.injEq]
  exact ⟨Fin.ext hr, Fin.ext hc, trivial⟩

/-! ### The tail

Six statements: the dead `c` binding, the two output coordinate vectors, the `C`
pointer tile, the two-axis mask, and the store — which writes `accumulator`, not
`c`, exactly as the source does. -/

/-! ## Main theorem -/

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4

#guard_msgs (drop info) in
#auditModuleAxioms
