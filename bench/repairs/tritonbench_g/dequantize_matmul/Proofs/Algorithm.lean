import bench.repairs.tritonbench_g.dequantize_matmul.Kernels
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.JIT
namespace VeriTile.Bench.Repairs.DequantizeMatmul.Algorithm
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
abbrev dequantize_kernel := TypedSource.kernel
/-! ### Row-major flattening of the 2-D tile into `⊨` lanes -/

/-- Lane `j` of the row-major flattening of an `R × C` tile. -/
def laneIdx (R C : Nat) (j : Fin (R * C)) : TileIndex [R, C] :=
  (⟨j.val / C, Nat.div_lt_of_lt_mul
      (Nat.lt_of_lt_of_le j.isLt (Nat.le_of_eq (Nat.mul_comm R C)))⟩,
   ⟨j.val % C, Nat.mod_lt _ (Nat.pos_of_ne_zero (by
      rintro rfl
      exact absurd j.isLt (by simp)))⟩,
   PUnit.unit)

/-- The lane of a logical `R × C` tile index — inverse of `laneIdx`. -/
def laneOf (R C : Nat) (idx : TileIndex [R, C]) : Fin (R * C) :=
  ⟨idx.1.val * C + idx.2.1.val, by
    have h1 : idx.1.val + 1 ≤ R := idx.1.isLt
    have h2 : idx.1.val * C + idx.2.1.val < idx.1.val * C + C :=
      Nat.add_lt_add_left idx.2.1.isLt _
    refine Nat.lt_of_lt_of_le h2 ?_
    rw [← Nat.succ_mul]
    exact Nat.mul_le_mul_right C h1⟩

@[simp] theorem laneIdx_laneOf (R C : Nat) (idx : TileIndex [R, C]) :
    laneIdx R C (laneOf R C idx) = idx := by
  obtain ⟨a, b, u⟩ := idx
  have hC : 0 < C := Nat.lt_of_le_of_lt (Nat.zero_le _) b.isLt
  have hdiv : (a.val * C + b.val) / C = a.val := by
    rw [Nat.mul_comm, Nat.mul_add_div hC, Nat.div_eq_of_lt b.isLt, Nat.add_zero]
  have hmod : (a.val * C + b.val) % C = b.val := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt b.isLt]
  simp only [laneIdx, laneOf, hdiv, hmod]

theorem laneOf_injective (R C : Nat) : Function.Injective (laneOf R C) :=
  Function.LeftInverse.injective (laneIdx_laneOf R C)

/-- Tile-index injectivity from the lane-indexed injectivity the `⊨` headline
states (`laneIdx` is onto the tile indices). -/
theorem inj_of_lane_inj {R C : Nat} {F : TileIndex [R, C] → Nat}
    (h : Function.Injective (fun j : Fin (R * C) => F (laneIdx R C j))) :
    Function.Injective F := by
  intro a b hab
  have hcomp : F (laneIdx R C (laneOf R C a)) = F (laneIdx R C (laneOf R C b)) := by
    rw [laneIdx_laneOf, laneIdx_laneOf]; exact hab
  exact laneOf_injective R C (h hcomp)

/-! ### Windows, gates and the per-lane spec -/

/-- Weight-tile read address of tile cell `idx` for program `(p₀, p₁)`. -/
def bOffset (p₀ p₁ stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (p₀ * BLOCK_SIZE_K + idx.1.val) * stride_bk +
    (p₁ * BLOCK_SIZE_N + idx.2.1.val) * stride_bn

/-- Output write address of tile cell `idx` for program `(p₀, p₁)`. -/
def fpbOffset (p₀ p₁ stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (p₀ * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
    (p₁ * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn

/-- Column-scale read address of tile column `j` for program `(_, p₁)`. -/
def nOffset (p₁ BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  p₁ * BLOCK_SIZE_N + j.val

/-- The kernel's store gate `mask` = `(k·BSK + i < K) & (n·BSN + j < N)`. -/
def dequantizeActive (p₀ p₁ K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Prop :=
  p₀ * BLOCK_SIZE_K + idx.1.val < K ∧ p₁ * BLOCK_SIZE_N + idx.2.1.val < N

instance dequantizeActiveDecidable
    (p₀ p₁ K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) :
    Decidable (dequantizeActive p₀ p₁ K N BLOCK_SIZE_N BLOCK_SIZE_K idx) := by
  unfold dequantizeActive
  infer_instance

/-- State-coupled readback spec (internal to `dequantize_kernel_correct`): the
exact dequantized value written at active tile lane `idx`. The `⊨` headline
re-expresses it as the pure product `xs j * ys j` of the loaded tiles. -/
noncomputable def dequantizeSpec
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (p₀ p₁ stride_bk stride_bn stride_scale BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : ℝ :=
  (s.readMemValue .int b_ptr (bOffset p₀ p₁ stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx) : ℝ) *
    s.readMem b_scale_ptr (nOffset p₁ BLOCK_SIZE_N idx.2.1 * stride_scale)

private noncomputable def dequantizeStoreBase
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (K N stride_bk stride_bn stride_scale stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    BlockState :=
  ((((((((((s.setReg "k_block_idx" .int [] (Tile.scalar (s.pids 0 : Int))).setReg
    "n_block_idx" .int [] (Tile.scalar (s.pids 1 : Int))).setReg
    "offs_k" .nat [BLOCK_SIZE_K] (Tile.vec fun i => i.val)).setReg
    "offs_n" .nat [BLOCK_SIZE_N] (Tile.vec fun i => i.val)).setReg
    "b_offs" .int [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => (bOffset (s.pids 0) (s.pids 1) stride_bk stride_bn
          BLOCK_SIZE_N BLOCK_SIZE_K i : Int) }).setReg
    "fpb_offs" .int [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => (fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn
          BLOCK_SIZE_N BLOCK_SIZE_K i : Int) }).setReg
    "bs_offs" .int [1, BLOCK_SIZE_N]
      { data := fun i => (nOffset (s.pids 1) BLOCK_SIZE_N i.2.1 * stride_scale : Nat) |> Int.ofNat }).setReg
    "n_mask" .bool [1, BLOCK_SIZE_N]
      { data := fun i => decide (nOffset (s.pids 1) BLOCK_SIZE_N i.2.1 < N) }).setReg
    "mask" .bool [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => decide (dequantizeActive (s.pids 0) (s.pids 1) K N
          BLOCK_SIZE_N BLOCK_SIZE_K i) }).setReg
    "int_b" .int [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i =>
          if dequantizeActive (s.pids 0) (s.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K i then
            s.readMemValue .int b_ptr (bOffset (s.pids 0) (s.pids 1) stride_bk stride_bn
              BLOCK_SIZE_N BLOCK_SIZE_K i)
          else (0 : Int) }).setReg
    "scale_b" .real [1, BLOCK_SIZE_N]
      { data := fun i =>
          if nOffset (s.pids 1) BLOCK_SIZE_N i.2.1 < N then
            some (s.readMem b_scale_ptr (nOffset (s.pids 1) BLOCK_SIZE_N i.2.1 * stride_scale))
          else some (0.0 : ℝ) }

/-- Direct execution of the actual typed source, for active lanes only.
The collision premise quantifies only active stores, so ragged tails are legal. -/
theorem source_active
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_scale stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s s' : BlockState)
    (hExec : exec (dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
          stride_scale stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K) s = some s')
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N])
    (hActive : dequantizeActive (s.pids 0) (s.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K idx)
    (hCollision : ∀ other,
      dequantizeActive (s.pids 0) (s.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K other →
      fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K other =
        fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K idx → other=idx) :
    s'.readMem fpb_ptr
      (fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K idx)=
      dequantizeSpec s b_ptr b_scale_ptr (s.pids 0) (s.pids 1) stride_bk stride_bn stride_scale
        BLOCK_SIZE_N BLOCK_SIZE_K idx := by
  simp [exec,dequantize_kernel,TypedSource.kernel, TypedSource.indexCommutedKernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeStmt.listToAlgorithm?,Except.bind,bind,pure,ComputeDType.eraseDType,stepStmts,stepStmt,evalOp.eq_def,Option.bind,
    Tile.bop,Tile.cop,Tile.expandDim,Tile.intToReal,NumericDType.add,NumericDType.mul,
    ComparableDType.lt,TileShape.dropInsertedIndex] at hExec
  simp only [←Int.natCast_mul, ←Int.natCast_add, Int.ofNat_lt, Int.toNat_natCast] at hExec
  rw [←hExec]
  let off : TileIndex [BLOCK_SIZE_K,BLOCK_SIZE_N] → Nat :=
    fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
  let active : TileIndex [BLOCK_SIZE_K,BLOCK_SIZE_N] → Prop :=
    dequantizeActive (s.pids 0) (s.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K
  let value : TileIndex [BLOCK_SIZE_K,BLOCK_SIZE_N] → ℝ := fun i => WithBot.unbotD 0
    (Option.map (fun scale => if active i then
      (s.readMemValue .int b_ptr (bOffset (s.pids 0) (s.pids 1) stride_bk stride_bn
        BLOCK_SIZE_N BLOCK_SIZE_K i) : ℝ)*scale else 0)
      (if nOffset (s.pids 1) BLOCK_SIZE_N i.2.1<N then
        some (s.readMem b_scale_ptr (nOffset (s.pids 1) BLOCK_SIZE_N i.2.1*stride_scale))
      else some (0.0 : ℝ)))
  have scatter := BlockState.scatter_readback_prop_masked_nd_of_true
    (region:=fpb_ptr)
    (dequantizeStoreBase s b_ptr b_scale_ptr K N stride_bk stride_bn stride_scale
      stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)
    off value active idx hActive hCollision
  rcases hActive with ⟨hk,hn⟩
  simpa [off,active,value,dequantizeActive,dequantizeSpec,bOffset,fpbOffset,nOffset,
    hk,hn,dequantizeStoreBase,TileShape.dropInsertedIndex,Bool.and_eq_true] using scatter

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `fpb_ptr`, and the masked-off
lanes of the output window itself — is preserved by the run. -/
theorem dequantize_kernel_frame
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_scale stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s s1 : BlockState)
    (hExec : exec ((dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn stride_scale
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N],
      dequantizeActive (s.pids 0) (s.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K idx →
      ¬(fpb_ptr = r ∧ fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn
          BLOCK_SIZE_N BLOCK_SIZE_K idx = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, dequantize_kernel, TypedSource.kernel, TypedSource.indexCommutedKernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeStmt.listToAlgorithm?,Except.bind,bind,pure,ComputeDType.eraseDType, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, Option.bind, ComputeExpr.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.intToReal,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex] at hExec
  simp only [←Int.natCast_mul, ←Int.natCast_add, Int.ofNat_lt, Int.toNat_natCast] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  refine hmiss k ?_ (by simpa [fpbOffset] using hc)
  simpa [dequantizeActive, Bool.and_eq_true] using hmk

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state (a
straight-line masked load / load / multiply / masked store). -/
theorem dequantize_kernel_exec_isSome
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_scale stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s : BlockState) :
    ∃ s1, exec ((dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn stride_scale
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).toAlgKernel) s = some s1 := by
  simp [exec, dequantize_kernel, TypedSource.kernel, TypedSource.indexCommutedKernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeStmt.listToAlgorithm?,Except.bind,bind,pure,ComputeDType.eraseDType, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, Option.bind, ComputeExpr.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.intToReal,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex]



/-- Only active output coordinates need be injective; masked tail addresses
may collide freely. Contiguous singleton strides are handled by Dense2. -/
theorem active_injective (K N BK BN p0 p1 : Nat) (dense : Dense2 K N)
    (i j : TileIndex [BK,BN])
    (hi : dequantizeActive p0 p1 K N BN BK i)
    (hj : dequantizeActive p0 p1 K N BN BK j)
    (same : fpbOffset p0 p1 dense.rowStride dense.colStride BN BK i=
      fpbOffset p0 p1 dense.rowStride dense.colStride BN BK j) : i=j := by
  let w : Window := ⟨0,0,K,N⟩
  have address : w.addr (p0*BK+i.1.val) (p1*BN+i.2.1.val)=
      w.addr (p0*BK+j.1.val) (p1*BN+j.2.1.val) := by
    apply Prod.ext
    · rfl
    · simpa only [Window.addr,w,Nat.zero_add,fpbOffset,
      dense.offset _ _ hi.1 hi.2,dense.offset _ _ hj.1 hj.2] using same
  have rows := congrArg w.row address
  have cols := congrArg w.col address
  have ci := w.coordinates (p0*BK+i.1.val) (p1*BN+i.2.1.val) hi.2
  have cj := w.coordinates (p0*BK+j.1.val) (p1*BN+j.2.1.val) hj.2
  rw [ci.1,cj.1] at rows
  rw [ci.2,cj.2] at cols
  apply Prod.ext
  · apply Fin.ext
    omega
  · apply Prod.ext
    · apply Fin.ext
      omega
    · exact Subsingleton.elim _ _

theorem source_complete (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_scale BN BK : Nat) (dense : Dense2 K N)
    (s : BlockState) :
    ∃ final,exec (dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
      stride_scale dense.rowStride dense.colStride BN BK) s=some final ∧
    (∀ idx : TileIndex [BK,BN],dequantizeActive (s.pids 0) (s.pids 1) K N BN BK idx →
      final.readMem fpb_ptr (fpbOffset (s.pids 0) (s.pids 1) dense.rowStride dense.colStride BN BK idx)=
      dequantizeSpec s b_ptr b_scale_ptr (s.pids 0) (s.pids 1) stride_bk stride_bn stride_scale BN BK idx) ∧
    (∀ region offset,
      (∀ idx : TileIndex [BK,BN],dequantizeActive (s.pids 0) (s.pids 1) K N BN BK idx →
        ¬(fpb_ptr=region ∧ fpbOffset (s.pids 0) (s.pids 1) dense.rowStride dense.colStride BN BK idx=offset)) →
      final.mem region offset=s.mem region offset) := by
  obtain ⟨final,executed⟩ := dequantize_kernel_exec_isSome b_ptr b_scale_ptr fpb_ptr
    K N stride_bk stride_bn stride_scale dense.rowStride dense.colStride BN BK s
  refine ⟨final,executed,?_,?_⟩
  · intro idx active
    apply source_active b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn stride_scale
      dense.rowStride dense.colStride BN BK s final executed idx active
    intro other otherActive same
    exact active_injective K N BK BN (s.pids 0) (s.pids 1) dense other idx otherActive active same
  · intro region offset outside
    exact dequantize_kernel_frame b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn stride_scale
      dense.rowStride dense.colStride BN BK s final executed region offset outside

#axiomsClean active_injective
#axiomsClean source_complete

#axiomsClean source_active
#axiomsClean dequantize_kernel_frame
#axiomsClean dequantize_kernel_exec_isSome
end
end VeriTile.Bench.Repairs.DequantizeMatmul.Algorithm
