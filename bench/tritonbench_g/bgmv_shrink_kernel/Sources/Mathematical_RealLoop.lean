import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Mathematical_RealFoundation
import VeriTile.Meta.StatementAudit

/-! Loop arithmetic with an explicit selected slot. The complete signed
metadata prefix must establish that slot and the invariant; this module
never substitutes natural metadata for the actual source load. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealLoop
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Input element `x[k] = input[cur_batch·xm_stride + k]` (upstream addresses
the input row directly by `current_k`; its `xk_stride` argument is unused). -/
noncomputable def aElem (s : BlockState) (input_ptr : RegionName)
    (xm_stride k : Nat) : ℝ :=
  s.readMem input_ptr (s.pids 1 * xm_stride + k)

/-- LoRA-A element
`W[n,k] = lora[l0_stride·lora_index + n·lora_k_stride + k·lora_n_stride]`. -/
noncomputable def bElem (s : BlockState) (lora_ptr : RegionName)
    (slot : Nat)
    (l0_stride lora_k_stride lora_n_stride n k : Nat) : ℝ :=
  s.readMem lora_ptr
    (l0_stride * slot + n * lora_k_stride
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
    (slot : Nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) : ℝ :=
  (Finset.univ : Finset (Fin BLOCK_K)).sum fun e =>
    if kIdx s BLOCK_K SPLIT_K c e.val < K then
      aElem s input_ptr xm_stride (kIdx s BLOCK_K SPLIT_K c e.val)
        * bElem s lora_ptr slot l0_stride lora_k_stride lora_n_stride
            n (kIdx s BLOCK_K SPLIT_K c e.val)
    else 0

/-- Partial accumulator after `c` K-split blocks at lane `n`. -/
noncomputable def accPartial (s : BlockState) (input_ptr lora_ptr : RegionName)
    (slot : Nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) : ℝ :=
  (Finset.range c).sum fun c' =>
    blockTerm s input_ptr lora_ptr slot K xm_stride l0_stride
      lora_k_stride lora_n_stride BLOCK_K SPLIT_K c' n

/-- One-block recurrence: `accPartial (c+1) = accPartial c + blockTerm c`. -/
theorem accPartial_succ (s : BlockState) (input_ptr lora_ptr : RegionName)
    (slot : Nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) :
    accPartial s input_ptr lora_ptr slot K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_K SPLIT_K (c + 1) n
      = accPartial s input_ptr lora_ptr slot K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_K SPLIT_K c n
        + blockTerm s input_ptr lora_ptr slot K xm_stride l0_stride
            lora_k_stride lora_n_stride BLOCK_K SPLIT_K c n := by
  unfold accPartial
  rw [Finset.sum_range_succ]

/-- Trip count of `for k in range(0, K, S)`: `⌈K/S⌉`. -/
def numKIters (K S : Nat) : Nat := (K + S - 1) / S

/-- This program's scaled rank-slice contribution at output
lane `n` — `scaling · Σ_{c<⌈K/S⌉} Σ_{e<BLOCK_K} [k(c,e)<K] · x[k(c,e)] ·
W[n,k(c,e)]`, a closed form over INPUT memory. -/
noncomputable def scaledContribution (s : BlockState) (input_ptr lora_ptr : RegionName)
    (slot : Nat) (scaling : ℝ)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K n : Nat) : ℝ :=
  scaling * accPartial s input_ptr lora_ptr slot K xm_stride l0_stride
    lora_k_stride lora_n_stride BLOCK_K SPLIT_K
    (numKIters K (BLOCK_K * SPLIT_K)) n

/-- **K-loop invariant** (counter `i = c · BLOCK_K·SPLIT_K`). After `c`
K-split blocks: pids fixed, the prologue registers seeded, memory untouched,
and `accumulator` holds the partial rank-slice sum `accPartial c` at active
lanes (`n < N`; inactive lanes hold `0` — their `tiled_b` loads are fully
masked). -/
noncomputable def shrinkInv
    (input_ptr lora_ptr : RegionName) (slot : Nat)
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
        ((lora_ptr, l0_stride * slot) : RegionName × Nat)) ∧
  s.regs .real [BLOCK_N] "accumulator"
    = some ⟨fun idx : TileIndex [BLOCK_N] =>
        some (if idx.1.val < N then
          accPartial s0 input_ptr lora_ptr slot K xm_stride l0_stride
            lora_k_stride lora_n_stride BLOCK_K SPLIT_K c idx.1.val
        else 0)⟩ ∧
  s.mem = s0.mem

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealLoop
