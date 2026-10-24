import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealProgram
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealTiles

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealDot
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 14000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def update (p : Parameters) : Op .real [p.BM,p.BN] :=
  .dotAcc .real .real .real (batch := [])
    (.ref .real [p.BM,p.BK] "a") (.ref .real [p.BK,p.BN] "b")
    (.ref .real [p.BM,p.BN] "accumulator")

def accumulator (p : Parameters) (xs : RealMath.Inputs)
    (row : Fin p.BM → Nat) (column : Fin p.BN → Nat) (ki : Nat) : Tile .real [p.BM,p.BN] :=
  Tile.mat fun i j => some (RealMath.partialSum p.K p.BK ki xs
    (RealTiles.clamp p.M (row i)) (RealTiles.clamp p.N (column j)))

/-- The update is the fused accumulator operation in the actual erased source,
not an independently substituted dot-plus-add program. -/
theorem matches_source (p : Parameters) : RealProgram.dot p =
    [.assign .real [p.BM,p.BN] "accumulator" (update p)] := by
  simp only [RealProgram.dot, RealProgram.body, RealSource.kernel,
    Parameters.source, TypedSource.kernel, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, FloatDType.toTileDType]
  rfl

/-- A genuine source dot step advances the independent input-only partial sum,
including its final masked K block. -/
theorem evaluates (p : Parameters) (s : BlockState) (xs : RealMath.Inputs)
    (row : Fin p.BM → Nat) (column : Fin p.BN → Nat) (ki : Nat)
    (ha : s.regs .real [p.BM,p.BK] "a" = some (RealTiles.aValues p xs row ki))
    (hb : s.regs .real [p.BK,p.BN] "b" = some (RealTiles.bValues p xs column ki))
    (hc : s.regs .real [p.BM,p.BN] "accumulator" = some (accumulator p xs row column ki)) :
    evalOp (update p) s = some (accumulator p xs row column (ki + 1)) := by
  simp only [update, evalOp.eq_def, ha, hb, hc, bind, Option.bind_some,
    FloatDType.cast, FloatDType.toTileDType, FloatDType.toWithBot, FloatDType.ofWithBot, List.nil_append]
  change some (⟨fun i => WithBot.realAdd
    ((Tile.dot [] (RealTiles.aValues p xs row ki) (RealTiles.bValues p xs column ki)).data i)
    ((accumulator p xs row column ki).data i)⟩ : Tile .real [p.BM,p.BN]) = _
  rw [RealTiles.masked_dot]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [accumulator, Tile.mat, WithBot.realAdd, RealMath.partial_succ]

theorem executes (p : Parameters) (s : BlockState) (xs : RealMath.Inputs)
    (row : Fin p.BM → Nat) (column : Fin p.BN → Nat) (ki : Nat)
    (ha : s.regs .real [p.BM,p.BK] "a" = some (RealTiles.aValues p xs row ki))
    (hb : s.regs .real [p.BK,p.BN] "b" = some (RealTiles.bValues p xs column ki))
    (hc : s.regs .real [p.BM,p.BN] "accumulator" = some (accumulator p xs row column ki)) :
    stepStmts (RealProgram.dot p) s = some
      (s.setReg "accumulator" .real [p.BM,p.BN] (accumulator p xs row column (ki+1))) := by
  rw [matches_source]
  simp only [stepStmts, stepStmt, evaluates p s xs row column ki ha hb hc, bind, Option.bind_some]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealDot
