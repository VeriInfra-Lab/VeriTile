import bench.tritonbench_g.dequantize_rowwise.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton
import bench.tritonbench_g.dequantize_matmul.Proofs.Mathematical_IntegerIO
import VeriTile.Meta.StatementAudit
/-! Mathematical output and full-cell frame lemmas for the actual source.
The kernel below is only its dtype-erased view; no alternate body is maintained.
Runtime scalar bindings and signed index conversions come from Kernels.lean. -/
namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise.RealReference
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
/-- Complete mathematical view of the canonical source, including its call ABI. -/
noncomputable def dequantize_rowwise_kernel
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat) : ComputeKernel :=
  (VeriTile.Bench.Repairs.DequantizeRowwise.source x_ptr state_x output_ptr inv_127
    n_elements BLOCK_SIZE P2).eraseDType

/-- Exact dequantized value written at active lane `i`. -/
noncomputable def dequantizeRowwiseSpec
    (s : BlockState) (x_ptr state_x : RegionName)
    (BLOCK_SIZE : Nat) (inv_127 : ℝ) (i : Fin P2) : ℝ :=
  s.readMem state_x s.pid *
    (s.readMemValue .int x_ptr (s.pid * BLOCK_SIZE + i.val) : ℝ) * inv_127

/-- Algorithm-layer correctness for `_dequantize_rowwise`.

The upstream kernel's active mask is `arange < BLOCK_SIZE`; `n_elements` is not
used by the kernel body. -/
theorem dequantize_rowwise_kernel_correct
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat)
    (s : BlockState) :
    ∀ i : Fin P2,
      let outAddr := s.pid * BLOCK_SIZE + i.val
      (exec (dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
            n_elements BLOCK_SIZE P2) s).map (·.readMem output_ptr outAddr)
        = some (if i.val < BLOCK_SIZE then
            dequantizeRowwiseSpec s x_ptr state_x BLOCK_SIZE inv_127 i
          else s.readMem output_ptr outAddr) := by
  intro i
  simp only [ComputeKernel.toAlgKernel_fromKernelBody,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeStmt.listToAlgorithm?_map_alg,dequantize_rowwise_kernel,
    VeriTile.Bench.Repairs.DequantizeRowwise.source,
    VeriTile.Bench.Repairs.DequantizeRowwise.body,
    VeriTile.Bench.Repairs.DequantizeRowwise.arguments,
    ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,ScalarArgument.integer,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeKernel.fromKernelBody,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    ComputeDType.eraseDType,FloatDType.toTileDType,eraseDType_float,Except.bind,bind,List.map_cons,List.map_nil,List.cons_append,List.nil_append]
  try simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,ComparableDType.eraseDType]
  simp [exec, dequantize_rowwise_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.intToReal, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : i.val < BLOCK_SIZE
  · simp [hi, dequantizeRowwiseSpec, BlockState.pid_eq, mul_assoc]
  · simp [hi]

set_option maxHeartbeats 4000000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `output_ptr`, and the *inactive* lanes of the output window itself. -/
private theorem dequantize_rowwise_kernel_frame
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat) (s s1 : BlockState)
    (hExec : exec ((dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
      n_elements BLOCK_SIZE P2).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin P2, i.val < BLOCK_SIZE →
      ¬(output_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp only [ComputeKernel.toAlgKernel_fromKernelBody,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeStmt.listToAlgorithm?_map_alg,dequantize_rowwise_kernel,
    VeriTile.Bench.Repairs.DequantizeRowwise.source,
    VeriTile.Bench.Repairs.DequantizeRowwise.body,
    VeriTile.Bench.Repairs.DequantizeRowwise.arguments,
    ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,ScalarArgument.integer,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeKernel.fromKernelBody,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    ComputeDType.eraseDType,FloatDType.toTileDType,eraseDType_float,Except.bind,bind,List.map_cons,List.map_nil,List.cons_append,List.nil_append] at hExec
  try simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,ComparableDType.eraseDType] at hExec
  simp [exec, dequantize_rowwise_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.intToReal, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose row
window holds `xs` and whose scale cell holds `ys` at the active lanes. This is
the `hrun` obligation of `Masked2DKernelIO₂.ImplementsInt.intro`; the value half
reuses `dequantize_rowwise_kernel_correct` (instantiated at the values the
state actually holds). -/
theorem dequantize_rowwise_kernel_region_run
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat)
    (s₀ : BlockState) (xs : Fin P2 → Int) (ys : Fin P2 → ℝ)
    (hx : ∀ j : Fin P2, j.val < BLOCK_SIZE →
      s₀.readMemValue .int x_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j)
    (hy : ∀ j : Fin P2, j.val < BLOCK_SIZE →
      s₀.readMem state_x s₀.pid = ys j) :
    ∃ s1, exec ((dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
        n_elements BLOCK_SIZE P2).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin P2, j.val < BLOCK_SIZE →
          s1.readMem output_ptr (s₀.pid * BLOCK_SIZE + j.val)
            = ys j * (xs j : ℝ) * inv_127)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin P2, j.val < BLOCK_SIZE →
            o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := dequantize_rowwise_kernel_correct x_ptr state_x output_ptr
    inv_127 n_elements BLOCK_SIZE P2 s₀
  rw [show exec (dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
      n_elements BLOCK_SIZE P2) s₀
      = exec ((dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
          n_elements BLOCK_SIZE P2).toAlgKernel) s₀ from rfl] at hobs
  cases hsrc : exec ((dequantize_rowwise_kernel x_ptr state_x output_ptr
      inv_127 n_elements BLOCK_SIZE P2).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp only [ComputeKernel.toAlgKernel_fromKernelBody,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeStmt.listToAlgorithm?_map_alg,dequantize_rowwise_kernel,
    VeriTile.Bench.Repairs.DequantizeRowwise.source,
    VeriTile.Bench.Repairs.DequantizeRowwise.body,
    VeriTile.Bench.Repairs.DequantizeRowwise.arguments,
    ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,ScalarArgument.integer,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeKernel.fromKernelBody,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    ComputeDType.eraseDType,FloatDType.toTileDType,eraseDType_float,Except.bind,bind,List.map_cons,List.map_nil,List.cons_append,List.nil_append]
        try simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,ComparableDType.eraseDType]
        simp [exec, dequantize_rowwise_kernel, ComputeKernel.toAlgKernel,
          stepStmts, stepStmt, evalOp.eq_def, Tile.intToReal, Tile.bop, Tile.cop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [Option.map_some, Option.some_inj, if_pos hj] at hje
        rw [hje]
        simp only [dequantizeRowwiseSpec]
        rw [hx j hj, hy j hj]
      · refine dequantize_rowwise_kernel_frame x_ptr state_x output_ptr
          inv_127 n_elements BLOCK_SIZE P2 s₀ s1 hsrc r o
          (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

#axiomsClean dequantize_rowwise_kernel_correct
#axiomsClean dequantize_rowwise_kernel_region_run
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeRowwise.RealReference
