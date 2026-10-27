import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardLoop
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def historicalLoop (p : BackwardLayout) (length : Nat) :=
  RealReference.diagSsmBackwardLoopBody p.s p.y p.gx p.gy length p.batch p.dim p.B

/-- Additional local facts established by the actual loop's first three
statements. No input or output oracle is assumed. -/
structure Indexed (p : BackwardLayout) (length iteration : Nat) (a b : BlockState) : Prop
    extends BackwardRelated p length a b where
  timeActual : a.regs .int [] "t" = some (Tile.scalar ((length - 1 - iteration : Nat) : Int))
  timeHistorical : b.regs .nat [] "t" = some (Tile.scalar (length - 1 - iteration))
  offsets : ∃ v : Tile .nat [p.B], a.regs .int [p.B] "offsets" = some ⟨fun i => (v.data i : Int)⟩ ∧
    b.regs .nat [p.B] "offsets" = some v
  outputGradient : ∃ v : Tile .real [p.B], a.regs .real [p.B] "grad_y" = some v ∧
    b.regs .real [p.B] "grad_y" = some v

theorem indexed_steps (p : BackwardLayout) (length t : Nat) (a b : BlockState)
    (h : BackwardRelated p length a b) (ht : t < length) :
    ∃ aa bb, stepStmts ((forwardLoop p length).take 3)
        (a.setReg "i" .int [] (Tile.scalar (t : Int))) = some aa ∧
      stepStmts ((historicalLoop p length).take 3)
        (b.setReg "i" .nat [] (Tile.scalar t)) = some bb ∧ Indexed p length t aa bb := by
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨state, hSa, hSb⟩, ⟨gradient, hGa, hGb⟩, ⟨transition, hLa, hLb⟩⟩
  have time_cast : (length : Int) - 1 - (t : Int) = (length - 1 - t : Nat) := by omega
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  rotate_left 2
  ·
    simp only [forwardLoop, forwardBody, Kernels.backwardTyped,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      ComputeDType.eraseDType, List.map_cons, List.map_nil,
      Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
    simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
    simp only [List.getElem?_cons_zero, List.getElem?_cons_succ, List.take]
    simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
      Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.gt, bind, Option.bind, Option.map_some]
    simp only [Tile.scalar, time_cast, Int.ofNat_lt, decide_true, decide_false,
      Bool.false_eq_true, String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
      hLength, hBatch, hDim, hCa, hMa, hSa, hGa, hLa, BlockState.readMemValue, Region.cast_cast]
    simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast, Nat.mul_assoc,
      BlockState.readMemAs, BlockState.readMem, memory, BlockState.writeMemTyped]
    rfl
  · simp only [historicalLoop, RealReference.diagSsmBackwardLoopBody, List.take,
      stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
      Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub, bind, Option.bind]
    simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, hCb, hMb,
      BlockState.readMemValue, BlockState.readMemAs, BlockState.readMem]
    rfl
  · try simp only [Tile.scalar, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast, Nat.mul_assoc]
    refine ⟨⟨rfl, pids, ?_, ?_, ?_, ⟨columns, ?_, ?_⟩, ⟨mask, ?_, ?_⟩,
      ⟨state, ?_, ?_⟩, ⟨gradient, ?_, ?_⟩, ⟨transition, ?_, ?_⟩⟩, ?_, ?_, ?_, ?_⟩
    all_goals simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, Tile.scalar,
      hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hGa, hGb, hLa, hLb]
    all_goals exact ⟨_, rfl, rfl⟩

#axiomsClean indexed_steps
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
