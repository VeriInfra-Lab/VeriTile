import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Mathematical foundation from the earlier active-surface proof.

The loop arithmetic, scalar evaluations and masked store/atomic readback
lemmas remain useful. Its bundled contracts concern separate active surfaces
with natural metadata; its flat IO covers the store surface. They do not
establish the complete signed typed-source contract now presented in
Correct.lean. The guarded surface here also has an independent SPLIT_K_ONE
flag. Kernels.lean retains the actual SPLIT_K == 1 comparison and every dtype,
cast, mask and the unused input-stride parameter.

RealLoop uses an explicit selected slot and RealSetup proves that the actual
signed prefix establishes it. RealProgram, RealSafety, RealFlat and RealIO
connect those lemmas to successful execution, independent input formulas,
active bounds and full-cell frames of both complete source versions. FP
proofs remain independent of this mathematical foundation.
-/

namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel

open VeriTile.Triton

set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-! ## Body decomposition (shared by both proof surfaces) -/

/-! ## Genuine closed-form spec -/

/-- The selected LoRA index (`lora_indices[cur_batch]`). -/
def loraIdx (s : BlockState) (lora_indices : Region .nat) : Nat :=
  s.readMemValue .nat lora_indices (s.pids 1)

/-- Input element `x[k] = input[cur_batch·xm_stride + k]` (upstream addresses
the input row directly by `current_k`; its `xk_stride` argument is unused). -/
noncomputable def aElem (s : BlockState) (input_ptr : RegionName)
    (xm_stride k : Nat) : ℝ :=
  s.readMem input_ptr (s.pids 1 * xm_stride + k)

/-- LoRA-A element
`W[n,k] = lora[l0_stride·lora_index + n·lora_k_stride + k·lora_n_stride]`. -/
noncomputable def bElem (s : BlockState) (lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (l0_stride lora_k_stride lora_n_stride n k : Nat) : ℝ :=
  s.readMem lora_ptr
    (l0_stride * loraIdx s lora_indices + n * lora_k_stride
      + k * lora_n_stride)

/-- The kernel's `current_k` rank index at loop iteration `c`, lane `e`:
`k(c,e) = c·(BLOCK_K·SPLIT_K) + (e + pid_sk·BLOCK_K)` — this program's
(`pid_sk`) slice of the rank dimension. -/
def kIdx (s : BlockState) (BLOCK_K SPLIT_K c e : Nat) : Nat :=
  c * (BLOCK_K * SPLIT_K) + (e + s.pids 0 * BLOCK_K)

/-- One K-split block's masked contribution at output lane `n`:
`Σ_{e<BLOCK_K} [k(c,e) < K] · x[k(c,e)] · W[n,k(c,e)]` (the `tl.sum` of the
masked `tiled_a · tiled_b` products; out-of-range rank lanes load `0`). -/
noncomputable def blockTerm (s : BlockState) (input_ptr lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) : ℝ :=
  (Finset.univ : Finset (Fin BLOCK_K)).sum fun e =>
    if kIdx s BLOCK_K SPLIT_K c e.val < K then
      aElem s input_ptr xm_stride (kIdx s BLOCK_K SPLIT_K c e.val)
        * bElem s lora_ptr lora_indices l0_stride lora_k_stride lora_n_stride
            n (kIdx s BLOCK_K SPLIT_K c e.val)
    else 0

/-- Partial accumulator after `c` K-split blocks at lane `n`. -/
noncomputable def accPartial (s : BlockState) (input_ptr lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) : ℝ :=
  (Finset.range c).sum fun c' =>
    blockTerm s input_ptr lora_ptr lora_indices K xm_stride l0_stride
      lora_k_stride lora_n_stride BLOCK_K SPLIT_K c' n

/-- One-block recurrence: `accPartial (c+1) = accPartial c + blockTerm c`. -/
theorem accPartial_succ (s : BlockState) (input_ptr lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) :
    accPartial s input_ptr lora_ptr lora_indices K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_K SPLIT_K (c + 1) n
      = accPartial s input_ptr lora_ptr lora_indices K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_K SPLIT_K c n
        + blockTerm s input_ptr lora_ptr lora_indices K xm_stride l0_stride
            lora_k_stride lora_n_stride BLOCK_K SPLIT_K c n := by
  unfold accPartial
  rw [Finset.sum_range_succ]

/-- Trip count of `for k in range(0, K, S)`: `⌈K/S⌉`. -/
def numKIters (K S : Nat) : Nat := (K + S - 1) / S

/-- **Genuine spec**: this program's scaled rank-slice contribution at output
lane `n` — `scaling · Σ_{c<⌈K/S⌉} Σ_{e<BLOCK_K} [k(c,e)<K] · x[k(c,e)] ·
W[n,k(c,e)]`, a closed form over INPUT memory. -/
@[standalone_spec] noncomputable def shrinkSpec (s : BlockState) (input_ptr lora_ptr : RegionName)
    (lora_indices : Region .nat) (scaling : ℝ)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K n : Nat) : ℝ :=
  scaling * accPartial s input_ptr lora_ptr lora_indices K xm_stride l0_stride
    lora_k_stride lora_n_stride BLOCK_K SPLIT_K
    (numKIters K (BLOCK_K * SPLIT_K)) n

/-- Output offset of lane `n`: `cur_batch·cm_stride + n·cn_stride`. -/
def outOff (s : BlockState) (cm_stride cn_stride n : Nat) : Nat :=
  s.pids 1 * cm_stride + n * cn_stride

/-- The ceil trip count sandwiches `K`: `K ≤ ⌈K/S⌉·S < K + S`. -/
theorem numKIters_bounds (K S : Nat) (hS : 0 < S) :
    K ≤ numKIters K S * S ∧ numKIters K S * S < K + S := by
  unfold numKIters
  have h := Nat.div_add_mod (K + S - 1) S
  have hm : (K + S - 1) % S < S := Nat.mod_lt _ hS
  rw [show (K + S - 1) / S * S = S * ((K + S - 1) / S) from Nat.mul_comm _ _]
  generalize S * ((K + S - 1) / S) = x at h ⊢
  omega

/-- Any counter with `K ≤ c·S < K + S` is the ceil trip count — pins the
`forRange_inv` exit counter to `numKIters`. -/
theorem kloop_final_iters (K S c : Nat) (hS : 0 < S)
    (h1 : K ≤ c * S) (h2 : c * S < K + S) : c = numKIters K S := by
  obtain ⟨hl, hr⟩ := numKIters_bounds K S hS
  rcases Nat.lt_trichotomy c (numKIters K S) with h | h | h
  · exfalso
    have hcs : c * S + S ≤ numKIters K S * S := by
      rw [show c * S + S = (c + 1) * S from (Nat.succ_mul c S).symm]
      exact Nat.mul_le_mul_right S h
    exact Nat.lt_irrefl _
      (lt_of_le_of_lt (le_trans (Nat.add_le_add_right h1 S) hcs) hr)
  · exact h
  · exfalso
    have hcs : numKIters K S * S + S ≤ c * S := by
      rw [show numKIters K S * S + S = (numKIters K S + 1) * S from
        (Nat.succ_mul _ S).symm]
      exact Nat.mul_le_mul_right S h
    exact Nat.lt_irrefl _
      (lt_of_le_of_lt (le_trans (Nat.add_le_add_right hl S) hcs) h2)

/-! ## Masked atomic-add scatter readback (generic `foldl` lemmas)

`stepStmt` reduces a masked `tl.atomic_add` to a lane `foldl` of per-cell
read-modify-writes. The shared library provides register/pid preservation for
this fold (`foldl_atomicAddAt_regs` / `foldl_atomicAddAt_pid`); the *memory
readback* lemmas below — the atomic analogue of `ScatterStore.foldl_store_at`
/ `foldl_store_preserve` — did not exist yet (this is the first bench consumer
of the atomic-add exec path), so they are proved here. -/

/-- A masked atomic-add `foldl` leaves an offset `o` untouched if no active
lane adds to it. -/
theorem foldl_atomicAdd_preserve {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (o : Nat) (l : List α)
    (s : BlockState) (hnot : ∀ k ∈ l, mask k → ofn k ≠ o) :
    (l.foldl (fun acc k =>
        if mask k then
          acc.writeMem region (ofn k) (acc.readMem region (ofn k) + vfn k)
        else acc) s).readMem region o = s.readMem region o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
    rw [List.foldl_cons]
    cases hm : mask hd
    · simp only [hm, Bool.false_eq_true, if_false]
      exact ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk)
    · simp only [hm, if_true]
      rw [ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk)]
      exact BlockState.writeMem_readMem_of_ne_offset s region (ofn hd) _ region o
        (hnot hd List.mem_cons_self (by rw [hm])).symm

/-- Readback at offset `O` of a masked atomic-add `foldl`: if a unique active
lane `a` adds to `O`, the readback is the **initial value plus its
contribution** — the per-program atomic-add obligation. -/
theorem foldl_atomicAdd_at {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (O : Nat) (l : List α)
    (s : BlockState) (a : α) (ha : a ∈ l) (hma : mask a) (hoa : ofn a = O)
    (huniq : ∀ b ∈ l, mask b → ofn b = O → b = a) (hnodup : l.Nodup) :
    (l.foldl (fun acc k =>
        if mask k then
          acc.writeMem region (ofn k) (acc.readMem region (ofn k) + vfn k)
        else acc) s).readMem region O
      = s.readMem region O + vfn a := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem ha
  subst hl
  rw [List.foldl_append, List.foldl_cons]
  rw [List.nodup_append, List.nodup_cons] at hnodup
  obtain ⟨hnd1, ⟨ha_notin2, hnd2⟩, hdisj⟩ := hnodup
  have h2 : ∀ b ∈ l₂, mask b → ofn b ≠ O := by
    intro b hb hmb heq
    exact ha_notin2 ((huniq b (by simp [List.mem_append, hb]) hmb heq) ▸ hb)
  have h1 : ∀ b ∈ l₁, mask b → ofn b ≠ O := by
    intro b hb hmb heq
    exact (hdisj b hb a List.mem_cons_self)
      (huniq b (by simp [List.mem_append, hb]) hmb heq)
  rw [foldl_atomicAdd_preserve ofn vfn mask O l₂ _ h2]
  simp only [hma, if_true, hoa]
  rw [BlockState.writeMem_readMem]
  rw [if_pos ⟨rfl, rfl⟩]
  rw [foldl_atomicAdd_preserve ofn vfn mask O l₁ _ h1]

/-! ## Prologue -/

/-! ## Loop invariant -/

/-- **K-loop invariant** (counter `i = c · BLOCK_K·SPLIT_K`). After `c`
K-split blocks: pids fixed, the prologue registers seeded, memory untouched,
and `accumulator` holds the partial rank-slice sum `accPartial c` at active
lanes (`n < N`; inactive lanes hold `0` — their `tiled_b` loads are fully
masked). -/
noncomputable def shrinkInv
    (input_ptr lora_ptr : RegionName) (lora_indices : Region .nat)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s0 : BlockState)
    (i : Nat) (s : BlockState) : Prop :=
  let c := i / (BLOCK_K * SPLIT_K)
  s.pids = s0.pids ∧ i = c * (BLOCK_K * SPLIT_K) ∧
  i < K + BLOCK_K * SPLIT_K ∧
  s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)) ∧
  s.regs .nat [BLOCK_N] "offset_n"
    = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) ∧
  s.regs .nat [BLOCK_K] "offset_k"
    = some (Tile.vec (fun e : Fin BLOCK_K => e.val + s0.pids 0 * BLOCK_K)) ∧
  s.regs .ptr [] "a_ptr"
    = some (Tile.scalar ((input_ptr, s0.pids 1 * xm_stride) : RegionName × Nat)) ∧
  s.regs .ptr [] "b_ptr"
    = some (Tile.scalar
        ((lora_ptr, l0_stride * loraIdx s0 lora_indices) : RegionName × Nat)) ∧
  s.regs .real [BLOCK_N] "accumulator"
    = some ⟨fun idx : TileIndex [BLOCK_N] =>
        some (if idx.1.val < N then
          accPartial s0 input_ptr lora_ptr lora_indices K xm_stride l0_stride
            lora_k_stride lora_n_stride BLOCK_K SPLIT_K c idx.1.val
        else 0)⟩ ∧
  s.mem = s0.mem

/-! ## Prologue eval helpers -/

theorem evalOp_ptrAdd' {a b shape} (bc : Broadcast a b shape)
    (ptr : Op .ptr a) (off : Op .nat b) (s : BlockState) :
    evalOp (.ptrAdd bc ptr off) s = (do
      let ptrs ← evalOp ptr s; let offs ← evalOp off s;
      some (Tile.ptrAdd bc ptrs offs)) := by simp [evalOp]

theorem evalOp_ptrBase' (region : RegionName) (s : BlockState) :
    evalOp (.ptrBase region) s = some (Tile.scalar (region.cast, 0)) := by
  simp [evalOp]

/-- `offset_k = tl.arange(0,BLOCK_K) + pid_sk·BLOCK_K` eval. -/
theorem offk_eval (BLOCK_K psk : Nat) (s : BlockState)
    (hsk : s.regs .nat [] "pid_sk" = some (Tile.scalar psk)) :
    evalOp (Op.add .nat Broadcast.scalarR (Op.arange BLOCK_K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_sk")
          (Op.constNat BLOCK_K))) s
      = some (Tile.vec (fun e : Fin BLOCK_K => e.val + psk * BLOCK_K)) := by
  have ha : evalOp (Op.arange BLOCK_K) s
      = some (Tile.vec (fun e : Fin BLOCK_K => e.val)) := by simp [Tile.vec]
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, ha, hsk,
    Option.bind_some, bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `a_ptr = input_ptr + cur_batch·xm_stride` eval. -/
theorem aptr_eval (input_ptr : RegionName) (xm_stride cb : Nat) (s : BlockState)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar cb)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase input_ptr)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch")
          (Op.constNat xm_stride))) s
      = some (Tile.scalar ((input_ptr, cb * xm_stride) : RegionName × Nat)) := by
  rw [evalOp_ptrAdd', evalOp_ptrBase']
  simp only [evalOp_mul, evalOp_ref, evalOp_constNat, hcb, Option.bind_some,
    bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro _
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, Region.cast]
  exact Prod.ext rfl (Nat.zero_add _)

/-- `b_ptr = lora_ptr + l0_stride·lora_index` eval. -/
theorem bptr_eval (lora_ptr : RegionName) (l0_stride li : Nat) (s : BlockState)
    (hli : s.regs .nat [] "lora_index" = some (Tile.scalar li)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase lora_ptr)
        (Op.mul .nat Broadcast.nil (Op.constNat l0_stride)
          (Op.ref .nat [] "lora_index"))) s
      = some (Tile.scalar ((lora_ptr, l0_stride * li) : RegionName × Nat)) := by
  rw [evalOp_ptrAdd', evalOp_ptrBase']
  simp only [evalOp_mul, evalOp_ref, evalOp_constNat, hli, Option.bind_some,
    bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro _
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, Region.cast]
  exact Prod.ext rfl (Nat.zero_add _)

/-- `accumulator = tl.zeros([BLOCK_N])` eval. -/
theorem acc_init_eval (BLOCK_N : Nat) (s : BlockState) :
    evalOp (Op.full [BLOCK_N] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_N]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-! ## Loop-body eval helpers -/

@[simp] theorem evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

@[simp] theorem evalOp_expandDim_one_nat {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] name)) s =
      (s.regs .nat [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- `current_k = k + offset_k` eval. -/
theorem currentK_eval (BLOCK_K i : Nat) (g : Fin BLOCK_K → Nat) (s : BlockState)
    (hk : s.regs .nat [] "k" = some (Tile.scalar i))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec g)) :
    evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "k")
        (Op.ref .nat [BLOCK_K] "offset_k")) s
      = some (Tile.vec (fun e : Fin BLOCK_K => i + g e)) := by
  simp only [evalOp_add, evalOp_ref, hk, hok, Option.bind]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add]

/-- `tiled_a = tl.load(a_ptr + current_k_c, mask=current_k < K, other=0.0)`
eval: lane `e` reads `input[apo + ck e]` when `ck e < K`, else `0`. -/
theorem tiledA_eval (input_ptr : RegionName) (K BLOCK_K apo : Nat)
    (ck : Fin BLOCK_K → Nat) (s0 s : BlockState)
    (hap : s.regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, apo) : RegionName × Nat)))
    (hckc : s.regs .nat [BLOCK_K] "current_k_c" = some (Tile.vec ck))
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck))
    (hrm : ∀ o, s.readMem input_ptr o = s0.readMem input_ptr o) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
            (Op.ref .nat [BLOCK_K] "current_k_c")))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "current_k")
            (Op.constNat K))
          ((Op.const 0.0).broadcast [BLOCK_K]))) s
      = some (⟨fun e : TileIndex [BLOCK_K] =>
          some (if ck e.1 < K then s0.readMem input_ptr (apo + ck e.1) else 0)⟩
            : Tile .real [BLOCK_K]) := by
  simp only [evalOp, evalOp.eq_def, evalOp_ref, hap, hckc, hck, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.scalar, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.cop, Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec]
  by_cases hkK : ck e.1 < K
  · simp only [hkK, decide_true, if_true, BlockState.readMemValue_real, hrm]
  · simp only [hkK, decide_false, Bool.false_eq_true, if_false]
    norm_num

/-- `b_ptr_mask = (offset_n[:,None] < N) & (current_k[None,:] < K)` eval. -/
theorem bmask_eval (N K BLOCK_N BLOCK_K : Nat) (ck : Fin BLOCK_K → Nat)
    (s : BlockState)
    (hon : s.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck)) :
    evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
          (Op.constNat N))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
          (Op.constNat K))) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          decide (idx.1.val < N) && decide (ck idx.2.1 < K)⟩
            : Tile .bool [BLOCK_N, BLOCK_K]) := by
  simp only [evalOp, evalOp.eq_def, evalOp_expandDim_one_nat, evalOp_expandDim_zero_nat,
    evalOp_ref, hon, hck, Option.bind_some, bind, Option.bind_eq_bind, Option.bind,
    evalOp_constNat]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.bop, Tile.cop, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]
  rfl

/-- `tiled_b` masked-load eval: lane `(n,e)` reads
`lora[bpo + n·lora_k_stride + ck e·lora_n_stride]` when both mask legs hold,
else `0`. -/
theorem tiledB_eval (lora_ptr : RegionName)
    (N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K bpo : Nat)
    (ck : Fin BLOCK_K → Nat) (s0 s : BlockState)
    (hbp : s.regs .ptr [] "b_ptr"
      = some (Tile.scalar ((lora_ptr, bpo) : RegionName × Nat)))
    (hon : s.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck))
    (hmask : s.regs .bool [BLOCK_N, BLOCK_K] "b_ptr_mask"
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          decide (idx.1.val < N) && decide (ck idx.2.1 < K)⟩
            : Tile .bool [BLOCK_N, BLOCK_K]))
    (hrm : ∀ o, s.readMem lora_ptr o = s0.readMem lora_ptr o) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consL.consR
            (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
                (Op.constNat lora_k_stride)))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
              (Op.constNat lora_n_stride))))
        (MaskOpt.maskOther (Op.ref .bool [BLOCK_N, BLOCK_K] "b_ptr_mask")
          ((Op.const 0.0).broadcast [BLOCK_N, BLOCK_K]))) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          some (if idx.1.val < N ∧ ck idx.2.1 < K then
            s0.readMem lora_ptr
              (bpo + idx.1.val * lora_k_stride + ck idx.2.1 * lora_n_stride)
          else 0)⟩ : Tile .real [BLOCK_N, BLOCK_K]) := by
  simp only [evalOp, evalOp.eq_def, evalOp_expandDim_one_nat, evalOp_expandDim_zero_nat,
    evalOp_ref, hbp, hon, hck, hmask, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.scalar, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, evalOp_constNat]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.cop, Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec,
    Tile.expandDim, TileShape.dropInsertedIndex]
  by_cases hn : idx.1.val < N
  · by_cases hkK : ck idx.2.1 < K
    · simp only [hn, hkK, decide_true, Bool.and_self, if_true, and_true, true_and,
        BlockState.readMemValue_real, hrm]
    · simp only [hn, hkK, decide_true, decide_false, Bool.and_false,
        Bool.false_eq_true, if_false, and_false]
      norm_num
  · simp only [hn, decide_false, Bool.false_and, Bool.false_eq_true, if_false,
      false_and]
    norm_num

/-- `accumulator = accumulator + tl.sum(tiled_a * tiled_b, 1)` eval. -/
theorem accadd_eval (BLOCK_N BLOCK_K : Nat) (za : Tile .real [BLOCK_N])
    (ta : Tile .real [BLOCK_K]) (tb : Tile .real [BLOCK_N, BLOCK_K])
    (s : BlockState)
    (hz : s.regs .real [BLOCK_N] "accumulator" = some za)
    (hta : s.regs .real [BLOCK_K] "tiled_a" = some ta)
    (htb : s.regs .real [BLOCK_N, BLOCK_K] "tiled_b" = some tb) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.ref .real [BLOCK_N] "accumulator")
        (Op.reduceSum ⟨1, by simp⟩ Bool.false
          (Op.mul .real Broadcast.nil.consSame.leadL
            (Op.ref .real [BLOCK_K] "tiled_a")
            (Op.ref .real [BLOCK_N, BLOCK_K] "tiled_b")))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame za
          (Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
            (Tile.bop NumericDType.real.mul Broadcast.nil.consSame.leadL ta tb))) := by
  have hrs : evalOp (Op.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (Op.mul .real Broadcast.nil.consSame.leadL
        (Op.ref .real [BLOCK_K] "tiled_a")
        (Op.ref .real [BLOCK_N, BLOCK_K] "tiled_b"))) s
      = some (Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
          (Tile.bop NumericDType.real.mul Broadcast.nil.consSame.leadL ta tb)) := by
    simp only [evalOp, evalOp_ref, hta, htb, Option.bind, Option.map]
    rfl
  rw [evalOp_add]
  simp only [evalOp_ref, hz, Option.bind_some, bind, Option.bind_eq_bind]
  erw [hrs]
  rfl

/-- **Per-lane reduction bridge**: the `tl.sum(tiled_a·tiled_b, 1)` of the two
masked loaded tiles at output lane `j` is the masked block contribution — `0`
whenever `j ≥ N` (the fully-masked `tiled_b` row). -/
theorem reduceSum_lane (N K BLOCK_N BLOCK_K : Nat) (ck : Fin BLOCK_K → Nat)
    (av : Nat → ℝ) (bv : Nat → Nat → ℝ) (j : Fin BLOCK_N) :
    (Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (Tile.bop NumericDType.real.mul Broadcast.nil.consSame.leadL
        (⟨fun e : TileIndex [BLOCK_K] =>
          some (if ck e.1 < K then av (ck e.1) else 0)⟩ : Tile .real [BLOCK_K])
        (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          some (if idx.1.val < N ∧ ck idx.2.1 < K then bv idx.1.val (ck idx.2.1)
            else 0)⟩ : Tile .real [BLOCK_N, BLOCK_K]))).data (j, PUnit.unit)
      = some (if j.val < N then
          (Finset.univ : Finset (Fin BLOCK_K)).sum fun e =>
            if ck e < K then av (ck e) * bv j.val (ck e) else 0
        else 0) := by
  simp only [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  have hterm : ∀ e : Fin BLOCK_K,
      NumericDType.real.mul (some (if ck e < K then av (ck e) else 0))
        (some (if j.val < N ∧ ck e < K then bv j.val (ck e) else 0))
      = ((if j.val < N ∧ ck e < K then av (ck e) * bv j.val (ck e) else 0 : ℝ)
          : WithBot ℝ) := by
    intro e
    by_cases hkK : ck e < K
    · by_cases hn : j.val < N
      · simp only [hkK, hn, true_and, and_self, if_true, NumericDType.mul,
          WithBot.realMul, Option.map₂, Option.bind, Option.map]
        rfl
      · simp only [hkK, hn, if_true, false_and, if_false, NumericDType.mul,
          WithBot.realMul, Option.map₂, Option.bind, Option.map]
        norm_num
        rfl
    · simp only [hkK, if_false, and_false, NumericDType.mul,
        WithBot.realMul, Option.map₂, Option.bind, Option.map]
      norm_num
      rfl
  rw [Finset.sum_congr rfl (fun e _ => hterm e), ← WithBot.coe_sum]
  by_cases hn : j.val < N
  · simp only [hn, true_and, if_true]
    rfl
  · simp only [hn, false_and, if_false, Finset.sum_const_zero]
    rfl

/-! ## Tail eval helpers -/

/-- `accumulator *= scaling` eval. -/
theorem scale_eval (BLOCK_N : Nat) (scaling : ℝ) (g : Fin BLOCK_N → ℝ)
    (s : BlockState)
    (hacc : s.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun idx : TileIndex [BLOCK_N] => some (g idx.1)⟩) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BLOCK_N] "accumulator")
        (Op.const scaling)) s
      = some (⟨fun idx : TileIndex [BLOCK_N] => some (g idx.1 * scaling)⟩
          : Tile .real [BLOCK_N]) := by
  simp only [evalOp_mul, evalOp_ref, evalOp_const, hacc, Option.bind_some, bind,
    Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]

/-- `c_ptr = out_ptr + cur_batch·cm_stride + offset_cn·cn_stride` eval. -/
theorem cptr_tail_eval (out_ptr : RegionName) (cm_stride cn_stride BLOCK_N cb : Nat)
    (s : BlockState)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar cb))
    (hocn : s.regs .nat [BLOCK_N] "offset_cn"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase out_ptr)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch")
            (Op.constNat cm_stride))
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
            (Op.constNat cn_stride)))) s
      = some (⟨fun idx : TileIndex [BLOCK_N] =>
          ((out_ptr, cb * cm_stride + idx.1.val * cn_stride) : RegionName × Nat)⟩
            : Tile .ptr [BLOCK_N]) := by
  rw [evalOp_ptrAdd', evalOp_ptrBase']
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, hcb, hocn,
    Option.bind_some, bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul, Region.cast]
  exact Prod.ext rfl (Nat.zero_add _)

/-- `c_mask = offset_cn < N` eval. -/
theorem cmask_eval (N BLOCK_N : Nat) (s : BlockState)
    (hocn : s.regs .nat [BLOCK_N] "offset_cn"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val))) :
    evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
        (Op.constNat N)) s
      = some (⟨fun idx : TileIndex [BLOCK_N] => decide (idx.1.val < N)⟩
          : Tile .bool [BLOCK_N]) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hocn, Option.bind_some, bind,
    Option.bind_eq_bind, Option.bind]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.cop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt]
  rfl

/-! ## Terminal statements: masked store / masked atomic add -/

/-! ## Exec-level correctness (both tail branches) -/

/-! ## Realizes_without_Rounding summaries -/

/-! ## Sentinel early-return path -/

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-! ## The `⊨[R]` streaming metadata headline (store face)

Everything below is purely additive; the exact surfaces and the main theorem
above are untouched. The `SPLIT_K = 1` store face is restated on the
metadata-slot streaming skin `StreamMetaMasked3DKernelIO₂`: one `.nat` slot
(`lora_indices[cur_batch]`, read at the pid-only cell `cur_batch`), two
masked streamed inputs with `other=0` (the `tiled_a` row tile and the
`tiled_b` weight tile, both gated by the genuine `current_k < K` tail mask),
and one masked terminal `.real` store. This kernel has **zero rounding
events** — every load, the whole K-loop arithmetic and the terminal store
are at `.real` — so under `execR R` the prologue, the loop and the common
tail collapse verbatim onto the exact stepper and the proven
`shrink_preLoop` / `shrink_step` / `forRange_inv` / `tail_common_steps`
stack is reused unchanged; only the single masked store statement is
re-proved on the `R` side, where the `.real` write coincides with the
`writeMemAsR R .real` rounded write (`RoundingModel.storeValue_real`) and
the readback contract's `R.round .real` is the identity (`round_real`) —
the `⊨[R]` face at the `.real` grid is the exact streaming contract, stated
once for every `R`.

The **atomic face** (`SPLIT_K > 1`, `tl.atomic_add`) is a read-modify-write,
not a terminal single-store fold, so it is *not* expressible in this skin's
S1 genre; its per-program `out-before + contribution` obligation stays with
the existing five-conjunct exact summary
`bgmv_shrink_kernel_output_summary_general` above (conjunct (5)). Likewise
the `-1` skip sentinel lives on the faithful `.int` surface
(`bgmv_shrink_sentinel_skip_no_write`, conjunct (3)); the store face below
follows the store proof surface's `Region .nat` metadata convention, the
sentinel skip being the trusted host boundary. -/

open scoped VeriTile.Triton.StreamMetaMasked3DKernelIO₂

/-! ### Cast-free collapses -/

/-! ### `evalOpR` collapses and eval helpers for the safety walk

The safety walk (`hts`) quantifies over arbitrary launch states, so it
re-evaluates the address/mask registers under `evalOpR R`. Every such op is
`R`-free (pure nat/bool/pointer arithmetic), so each collapses onto `evalOp`
and reuses the exact eval lemmas above. -/

private theorem evalOpR_programId (R : RoundingModel) (ax : Nat)
    (s : BlockState) :
    evalOpR R (Op.programId ax) s = some (Tile.scalar (s.pids ax)) := by
  simp only [evalOpR]

private theorem evalOpR_arange (R : RoundingModel) (n : Nat) (s : BlockState) :
    evalOpR R (Op.arange n) s
      = some (Tile.vec (fun j : Fin n => j.val)) := by
  simp only [evalOpR, Tile.vec]

/-- `current_k = k + offset_k` is `R`-free. -/
private theorem evalR_currentK (R : RoundingModel) (BLOCK_K : Nat)
    (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "k")
        (Op.ref .nat [BLOCK_K] "offset_k")) s
      = evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "k")
          (Op.ref .nat [BLOCK_K] "offset_k")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `tiled_a` load mask `current_k < K` is `R`-free. -/
private theorem evalR_maskA (R : RoundingModel) (K BLOCK_K : Nat)
    (s : BlockState) :
    evalOpR R (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "current_k")
        (Op.constNat K)) s
      = evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "current_k")
          (Op.constNat K)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `tiled_a` pointer `a_ptr + current_k_c` is `R`-free. -/
private theorem evalR_aTilePtr (R : RoundingModel) (BLOCK_K : Nat)
    (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
        (Op.ref .nat [BLOCK_K] "current_k_c")) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
          (Op.ref .nat [BLOCK_K] "current_k_c")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `b_ptr_mask` op is `R`-free. -/
private theorem evalR_bmask (R : RoundingModel) (N K BLOCK_N BLOCK_K : Nat)
    (s : BlockState) :
    evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
          (Op.constNat N))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
          (Op.constNat K))) s
      = evalOp (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
            (Op.constNat N))
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
            (Op.constNat K))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `tiled_b` pointer op is `R`-free. -/
private theorem evalR_bTilePtr (R : RoundingModel)
    (lora_k_stride lora_n_stride BLOCK_N BLOCK_K : Nat) (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
            (Op.constNat lora_k_stride)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
          (Op.constNat lora_n_stride))) s
      = evalOp (Op.ptrAdd Broadcast.nil.consL.consR
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
              (Op.constNat lora_k_stride)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
            (Op.constNat lora_n_stride))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `c_ptr` tail op is `R`-free. -/
private theorem evalR_cptr (R : RoundingModel) (out_ptr : RegionName)
    (cm_stride cn_stride BLOCK_N : Nat) (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase out_ptr)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch")
            (Op.constNat cm_stride))
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
            (Op.constNat cn_stride)))) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase out_ptr)
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch")
              (Op.constNat cm_stride))
            (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
              (Op.constNat cn_stride)))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `c_mask` tail op is `R`-free. -/
private theorem evalR_cmask (R : RoundingModel) (N BLOCK_N : Nat)
    (s : BlockState) :
    evalOpR R (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
        (Op.constNat N)) s
      = evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_cn")
          (Op.constNat N)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- `current_k < K` mask eval (the 1-D sibling of `cmask_eval`). -/
private theorem maskA_eval (K BLOCK_K : Nat) (ck : Fin BLOCK_K → Nat)
    (s : BlockState)
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck)) :
    evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "current_k")
        (Op.constNat K)) s
      = some (⟨fun e : TileIndex [BLOCK_K] => decide (ck e.1 < K)⟩
          : Tile .bool [BLOCK_K]) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hck, Option.bind_some, bind,
    Option.bind_eq_bind, Option.bind]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.cop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt]
  rfl

/-- `a_ptr + current_k_c` pointer-tile eval. -/
private theorem aTilePtr_eval (input_ptr : RegionName) (BLOCK_K apo : Nat)
    (ck : Fin BLOCK_K → Nat) (s : BlockState)
    (hap : s.regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, apo) : RegionName × Nat)))
    (hckc : s.regs .nat [BLOCK_K] "current_k_c" = some (Tile.vec ck)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
        (Op.ref .nat [BLOCK_K] "current_k_c")) s
      = some (⟨fun e : TileIndex [BLOCK_K] =>
          ((input_ptr, apo + ck e.1) : RegionName × Nat)⟩
            : Tile .ptr [BLOCK_K]) := by
  rw [evalOp_ptrAdd']
  simp only [evalOp_ref, hap, hckc, Option.bind_some, bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add]

/-- `tiled_b` pointer-tile eval: cell `(n, e)` addresses
`bpo + n·lora_k_stride + ck e·lora_n_stride` in the LoRA buffer. -/
private theorem bTilePtr_eval (lora_ptr : RegionName)
    (lora_k_stride lora_n_stride BLOCK_N BLOCK_K bpo : Nat)
    (ck : Fin BLOCK_K → Nat) (s : BlockState)
    (hbp : s.regs .ptr [] "b_ptr"
      = some (Tile.scalar ((lora_ptr, bpo) : RegionName × Nat)))
    (hon : s.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck)) :
    evalOp (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
            (Op.constNat lora_k_stride)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
          (Op.constNat lora_n_stride))) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          ((lora_ptr, bpo + idx.1.val * lora_k_stride
              + ck idx.2.1 * lora_n_stride) : RegionName × Nat)⟩
            : Tile .ptr [BLOCK_N, BLOCK_K]) := by
  simp only [evalOp, evalOp.eq_def, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, evalOp_ref, evalOp_constNat, hbp, hon, hck,
    Option.bind, Option.map, Option.bind_some, bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, NumericDType.add]
  rfl

/-! ### The masked `.real` store under `execR R` -/

/-- A `.real`-typed rounded write **is** the `writeMemAsR R .real` write
(`writeMemTypedR` delegates `.real` to the exact `writeMemTyped`, and
`RoundingModel.storeValue_real` says `R.storeValue .real` is exactly the
finite-fallback demotion). Lets the `.real` masked store reuse the
`writeMemAsR` scatter readback/frame family. -/
private theorem writeMemTypedR_real_eq (R : RoundingModel) (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier TileDType.real) :
    s.writeMemTypedR R .real region offset v
      = s.writeMemAsR R .real region offset v := by
  show s.writeMemTyped .real region offset v = _
  simp only [BlockState.writeMemTyped, BlockState.writeMemAs,
    BlockState.writeMemAsR, RoundingModel.storeValue_real]

/-- Tag-exact readback of a stored `.real` cell through `readMemAs .real`:
the `storeValue ∘ ofReal` round trip is the identity on a defined real. -/
private theorem readMemAs_real_of_cell {s : BlockState} {region : RegionName}
    {offset : Nat} {x : ℝ}
    (h : s.mem region offset
      = MemCell.of FloatDType.real.toTileDType (FloatDType.real.ofReal x)) :
    s.readMemAs .real region offset = FloatDType.real.ofReal x := by
  simp [BlockState.readMemAs, h, FloatDType.storeValue, FloatDType.ofReal]

/-- The output-window scatter offsets are injective over the whole lane tile
(from `0 < cn_stride`, via `affine1D_inj`). -/
private theorem outOff_tileInj (s : BlockState) (cm_stride cn_stride : Nat)
    (hcn : 0 < cn_stride) {BLOCK_N : Nat} :
    Function.Injective (fun idx : TileIndex [BLOCK_N] =>
      outOff s cm_stride cn_stride idx.1.val) := by
  intro a b hab
  have h1 : a.1 = b.1 :=
    affine1D_inj (s.pids 1 * cm_stride) cn_stride hcn
      (show (fun k : Fin BLOCK_N => s.pids 1 * cm_stride + k.val * cn_stride) a.1
          = (fun k : Fin BLOCK_N => s.pids 1 * cm_stride + k.val * cn_stride) b.1
        from hab)
  exact Prod.ext h1 rfl

/-! ### The `TraceSafeR` safety walk -/

/-- The loop counter of an in-range iteration sits below the ceil trip count:
`i < K` with `i = (i/S)·S` forces `i/S < ⌈K/S⌉`. -/
private theorem kloop_counter_lt (K S i : Nat) (hS : 0 < S) (hilt : i < K)
    (hi : i = i / S * S) : i / S < numKIters K S := by
  by_contra h
  have h1 : numKIters K S * S ≤ i / S * S :=
    Nat.mul_le_mul_right S (Nat.le_of_not_lt h)
  have h2 : K ≤ i / S * S := le_trans (numKIters_bounds K S hS).1 h1
  rw [← hi] at h2
  exact absurd hilt (Nat.not_lt.mpr h2)

/-! ### IO signature, lane bridge, spec bridge -/

/-- The `tiled_b`-stream lane feeding output lane `j` at rank key `e`: `(j, e)`
row-major over the `[BLOCK_N, BLOCK_K]` per-step weight tile, via the shared
`Lane2D` bridge. -/
def bStreamLane (BLOCK_N BLOCK_K : Nat) (j : Fin BLOCK_N) (e : Fin BLOCK_K) :
    Fin (BLOCK_N * BLOCK_K) :=
  Lane2D.encode (j, e, PUnit.unit)

/-- Under the slot pin and the two stream pins, the exact stack's
`accPartial` at the full trip count **is** the skin-level masked double fold
`∑ t, ∑ e, [current_k < K] · xs·ys` (per write-active lane `j < N`; the
`tiled_b` pins are addressed through the loaded slot value `m0`). -/
private theorem shrinkSpec_eq_streamSum
    (input_ptr lora_ptr : RegionName) (lora_indices : Region .nat)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s₀ : BlockState) (m0 : Nat)
    (hli : loraIdx s₀ lora_indices = m0)
    (xs : Fin (numKIters K (BLOCK_K * SPLIT_K)) → Fin BLOCK_K → ℝ)
    (ys : Fin (numKIters K (BLOCK_K * SPLIT_K)) → Fin (BLOCK_N * BLOCK_K) → ℝ)
    (hx : ∀ (t : Fin (numKIters K (BLOCK_K * SPLIT_K))) (e : Fin BLOCK_K),
      t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K) < K →
      s₀.readMem input_ptr (s₀.pids 1 * xm_stride
          + (t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K)))
        = xs t e)
    (hy : ∀ (t : Fin (numKIters K (BLOCK_K * SPLIT_K)))
        (l : Fin (BLOCK_N * BLOCK_K)),
      l.val / BLOCK_K < N ∧
        t.val * (BLOCK_K * SPLIT_K) + (l.val % BLOCK_K + s₀.pids 0 * BLOCK_K) < K →
      s₀.readMem lora_ptr (l0_stride * m0 + l.val / BLOCK_K * lora_k_stride
          + (t.val * (BLOCK_K * SPLIT_K)
              + (l.val % BLOCK_K + s₀.pids 0 * BLOCK_K)) * lora_n_stride)
        = ys t l)
    (j : Fin BLOCK_N) (hj : j.val < N) :
    accPartial s₀ input_ptr lora_ptr lora_indices K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_K SPLIT_K
        (numKIters K (BLOCK_K * SPLIT_K)) j.val
      = ∑ t : Fin (numKIters K (BLOCK_K * SPLIT_K)), ∑ e : Fin BLOCK_K,
          if t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K) < K
          then xs t e * ys t (bStreamLane BLOCK_N BLOCK_K j e)
          else 0 := by
  unfold accPartial
  rw [← Fin.sum_univ_eq_sum_range]
  refine Finset.sum_congr rfl fun t _ => ?_
  unfold blockTerm
  refine Finset.sum_congr rfl fun e _ => ?_
  have hkidx : kIdx s₀ BLOCK_K SPLIT_K t.val e.val
      = t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K) := rfl
  rw [hkidx]
  by_cases hkK : t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K) < K
  · rw [if_pos hkK, if_pos hkK]
    have hxa : aElem s₀ input_ptr xm_stride
        (t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K))
        = xs t e := hx t e hkK
    have hyb : bElem s₀ lora_ptr lora_indices l0_stride lora_k_stride
        lora_n_stride j.val
        (t.val * (BLOCK_K * SPLIT_K) + (e.val + s₀.pids 0 * BLOCK_K))
        = ys t (bStreamLane BLOCK_N BLOCK_K j e) := by
      rw [← hy t (bStreamLane BLOCK_N BLOCK_K j e)
          (by
            unfold bStreamLane
            rw [Lane2D.encode_div, Lane2D.encode_mod]
            exact ⟨hj, hkK⟩)]
      unfold bElem bStreamLane
      rw [Lane2D.encode_div, Lane2D.encode_mod, hli]
    rw [hxa, hyb]
  · rw [if_neg hkK, if_neg hkK]

/-! ### ════════ ★ STREAMING METADATA HEADLINE (store face) ★ ════════ -/

end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel
