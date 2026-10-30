import bench.repairs.tritonbench_g.diag_ssm_triton.Kernels
import bench.tritonbench_g.diag_ssm_triton.Proofs.Erasure
import bench.tritonbench_g.diag_ssm_triton.Proofs.RealReference
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def forwardBody (p : BackwardLayout) (length : Nat) :=
  (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType.toAlgKernel.body

def forwardPrefix (p : BackwardLayout) (length : Nat) := (forwardBody p length).take 9

def forwardPrelude (p : BackwardLayout) (length : Nat) (u : BlockState) :
    {v : BlockState // stepStmts (forwardPrefix p length) u = some v} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [forwardPrefix, forwardBody, Kernels.backwardTyped, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [List.take]
  simp only [stepStmts, stepStmt, evalOp, BlockState.setReg, Tile.scalar, Tile.bop, Tile.vec,
    NumericDType.add, NumericDType.mul, IntegralDType.mod, ComparableDType.lt, Option.bind, bind]
  simp
  try rfl

/-- The new prefix reads the source inputs without modifying memory. -/
theorem forwardPrelude_memory (p : BackwardLayout) (length : Nat) (u : BlockState) :
    (forwardPrelude p length u).val.mem = u.mem := by
  rfl

theorem forwardPrelude_pids (p : BackwardLayout) (length : Nat) (u : BlockState) :
    (forwardPrelude p length u).val.pids = u.pids := by
  rfl

theorem forwardPrelude_length (p : BackwardLayout) (length : Nat) (u : BlockState) :
    (forwardPrelude p length u).val.regs .int [] "length" =
      some (Tile.scalar (length : Int)) := by
  simp [forwardPrelude, BlockState.setReg]

def columns (p : BackwardLayout) (u : BlockState) (i : TileIndex [p.B]) : Nat :=
  u.pids 0 * p.B + i.1.val

theorem forwardPrelude_indices (p : BackwardLayout) (length : Nat) (u : BlockState) :
    (forwardPrelude p length u).val.regs .int [] "batch_size" = some (Tile.scalar (p.batch : Int)) ∧
    (forwardPrelude p length u).val.regs .int [] "dim" = some (Tile.scalar (p.dim : Int)) ∧
    (forwardPrelude p length u).val.regs .int [p.B] "col_offsets" =
      some ⟨fun i => (columns p u i : Int)⟩ ∧
    (forwardPrelude p length u).val.regs .bool [p.B] "mask" =
      some ⟨fun i => decide (columns p u i < p.batch * p.dim)⟩ := by
  refine ⟨rfl, rfl, ?_, ?_⟩
  · simp only [forwardPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
    change some ({data := fun i => (u.pids 0 : Int) * (p.B : Int) + (i.1.val : Int)} : Tile .int [p.B]) = _
    congr 1
  · simp only [forwardPrelude, BlockState.setReg, ↓reduceDIte, ↓reduceIte, castTile_self]
    change some ({data := fun i => decide (((u.pids 0 : Int) * (p.B : Int) + (i.1.val : Int)) < (p.batch : Int) * (p.dim : Int))} : Tile .bool [p.B]) = _
    congr 1
    ext i
    simp only [columns, ← Int.natCast_mul, ← Int.natCast_add, Int.ofNat_lt]
    rfl

/-- The existing unsigned mathematical prelude, used only for the semantic
connection to its independent recurrence proof. -/
def unsignedPrelude (p : BackwardLayout) (u : BlockState) :
    {v : BlockState // stepStmts (RealReference.diagSsmBackwardPreLoop p.l p.batch p.dim p.B) u = some v} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [RealReference.diagSsmBackwardPreLoop, stepStmts, stepStmt, evalOp,
    BlockState.setReg, Tile.scalar, Tile.bop, Tile.vec, NumericDType.add,
    NumericDType.mul, IntegralDType.mod, ComparableDType.lt, bind, Option.bind]
  simp
  try rfl

#axiomsClean forwardPrelude
#axiomsClean forwardPrelude_memory
#axiomsClean forwardPrelude_pids
#axiomsClean forwardPrelude_length
#axiomsClean forwardPrelude_indices
#axiomsClean unsignedPrelude
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
