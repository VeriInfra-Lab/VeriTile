import bench.tritonbench_g.triton_softmax.Kernels
import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.TritonSoftmax
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Masked input row tile used by `softmax_kernel`: lane `j < n_cols` holds
`xs j`, masked lanes are `⊥`, matching `other=-float("inf")`. -/
noncomputable def softmaxInputTile (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx => if idx.1.val < n_cols then some (xs idx.1) else none }

/-- Exact stable-softmax value computed by the kernel at lane `idx`, as a pure
function of the active row prefix `xs j`, `j < n_cols` (masked lanes enter the
reductions as `⊥`, neutral for both `max` and the `exp`-sum). -/
noncomputable def softmaxSpec (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile n_cols BLOCK_SIZE xs
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (idx, PUnit.unit))
  | none => 0

/-- `softmaxSpec` only reads the active lanes of its input: two rows agreeing
below `n_cols` yield the same softmax value (masked lanes are `⊥` in the tile
either way). -/
theorem softmaxSpec_congr (n_cols BLOCK_SIZE : Nat) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → xs j = ys j) (i : Fin BLOCK_SIZE) :
    softmaxSpec n_cols BLOCK_SIZE xs i = softmaxSpec n_cols BLOCK_SIZE ys i := by
  have htile : softmaxInputTile n_cols BLOCK_SIZE xs
      = softmaxInputTile n_cols BLOCK_SIZE ys := by
    unfold softmaxInputTile
    congr 1
    funext idx
    by_cases hj : idx.1.val < n_cols
    · simp only [if_pos hj, h idx.1 hj]
    · simp only [if_neg hj]
  unfold softmaxSpec
  rw [htile]

/-- Algorithm-layer cellwise correctness for `softmax_kernel`: in-bounds lanes
hold `softmaxSpec` of the loaded row, out-of-bounds lanes are preserved. -/
theorem softmax_kernel_correct
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
          n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := s.pid * output_row_stride + i.val
      s'.readMem output_ptr outAddr =
        if i.val < n_cols then
          softmaxSpec n_cols BLOCK_SIZE
            (fun j => s.readMem input_ptr (s.pid * input_row_stride + j.val)) i
        else s.readMem output_ptr outAddr := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, softmax_kernel, stepStmts, stepStmt, evalOp.eq_def,
          tile_elementwise, hB] at hExec
    subst s'
    simp [BlockState.pid_eq]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, softmaxSpec, softmaxInputTile, Tile.reduceMax, Tile.reduceMaxDrop,
            Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex, hB]
      congr
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `output_ptr`, and the *inactive*
lanes of the output row itself — is preserved by the run. -/
private theorem softmax_kernel_frame
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s s1 : BlockState)
    (hExec : exec ((softmax_kernel output_ptr input_ptr input_row_stride
        output_row_stride n_cols BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(output_ptr = r ∧ s.pid * output_row_stride + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, softmax_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, tile_elementwise, hB] at hExec
  subst s1
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state. `0 <
BLOCK_SIZE` is required because the `max` reduce (like `Finset.sup'`) is only
defined on non-empty axes. -/
private theorem softmax_kernel_exec_isSome
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s : BlockState) :
    ∃ s1, exec ((softmax_kernel output_ptr input_ptr input_row_stride
        output_row_stride n_cols BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, softmax_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, tile_elementwise, hB]

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input row is loaded at the **active lanes only** (`j < n_cols`). This is the
`hrun` obligation of the `⊨` headline. -/
theorem softmax_kernel_region_run
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem input_ptr (s₀.pid * input_row_stride + j.val) = xs j) :
    ∃ s1, exec ((softmax_kernel output_ptr input_ptr input_row_stride
          output_row_stride n_cols BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem output_ptr (s₀.pid * output_row_stride + j.val)
            = softmaxSpec n_cols BLOCK_SIZE xs j)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * output_row_stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := softmax_kernel_exec_isSome output_ptr input_ptr
    input_row_stride output_row_stride n_cols BLOCK_SIZE hB s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := softmax_kernel_correct output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE s₀ s1 hs1 j
    simp only [hj, if_pos] at h
    rw [h]
    exact softmaxSpec_congr n_cols BLOCK_SIZE _ xs (fun k hk => hx k hk) j
  · refine softmax_kernel_frame output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE hB s₀ s1 hs1 r o
      (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked load with `other`, reductions, masked store). -/
theorem softmax_kernel_flattenOk
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ((softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all nine
statements — seven are memory-silent (`program_id`, the pointer staging, the
reductions, and the register arithmetic) — and reduces the two masked accesses
(row load, row store) to the **lane-wise** bounds hypotheses: every *active*
lane's address is below the region bound. -/
theorem softmax_kernel_traceSafe
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * input_row_stride + j.val < bounds input_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * output_row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout
  simp [softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, tile_elementwise,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

end VeriTile.Bench.TritonBenchG.TritonSoftmax
