import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoads
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealFinish

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealStep
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def code (p : Parameters) := RealProgram.loads p ++ RealProgram.dot p ++ RealProgram.store p

def values (p : Parameters) (xs : RealMath.Inputs) (row column ki : Nat) :
    Fin p.BM → Fin p.BN → ℝ := fun i j =>
  RealMath.partialSum p.K p.BK (ki+1) xs
    (RealTiles.clamp p.M (RealStore.rows p row i))
    (RealTiles.clamp p.N (RealStore.columns p column j))

def middle (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (row column ki : Nat) : BlockState :=
  (RealLoads.result p s (RealStore.rows p row) (RealStore.columns p column) ki).setReg
    "accumulator" .real [p.BM,p.BN]
    (RealDot.accumulator p xs (RealStore.rows p row) (RealStore.columns p column) (ki+1))

def result (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (row column ki : Nat) : BlockState :=
  let t := middle p s xs row column ki
  if ki+1=RealMath.blocks p.K p.BK then RealStore.result p t row column (values p xs row column ki) else t

theorem matches_source (p : Parameters) : (RealProgram.body p).drop 2 = code p := by
  simp only [RealProgram.body,RealProgram.loads,RealProgram.dot,RealProgram.store,code,RealSource.kernel,
    Parameters.source, TypedSource.kernel, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, FloatDType.toTileDType]
  rfl

/-- Complete actual load-dot-completion execution from the current source
schedule state. The only numerical invariant is the independent partial input
sum; every load, fused update, branch, store and reset is proved operationally. -/
theorem executes (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (row column ki : Nat)
    (hM : 0 < p.M) (hN : 0 < p.N) (hinput : RealInput.Represents p xs s)
    (hk : s.regs .int [] "ki" = some (Tile.scalar (ki : Int)))
    (hcount : s.regs .nat [] "k_tiles" = some (Tile.scalar (RealMath.blocks p.K p.BK)))
    (hr : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hc : s.regs .int [] "pid_n" = some (Tile.scalar (column : Int)))
    (hm : s.regs .int [p.BM] "offs_am" = some (Tile.vec fun i =>
      (RealTiles.clamp p.M (RealStore.rows p row i) : Int)))
    (hn : s.regs .int [p.BN] "offs_bn" = some (Tile.vec fun j =>
      (RealTiles.clamp p.N (RealStore.columns p column j) : Int)))
    (hl : s.regs .nat [p.BK] "offs_k_for_mask" = some (Tile.vec Fin.val))
    (ha : s.regs .real [p.BM,p.BN] "accumulator" = some
      (RealDot.accumulator p xs (RealStore.rows p row) (RealStore.columns p column) ki)) :
    stepStmts ((RealProgram.body p).drop 2) s = some (result p s xs row column ki) := by
  let l := RealLoads.result p s (RealStore.rows p row) (RealStore.columns p column) ki
  have h1 := RealLoads.executes p s (RealStore.rows p row) (RealStore.columns p column) ki hk hm hn hl
  have h2 : stepStmts (RealProgram.dot p) l = some (middle p s xs row column ki) :=
    RealDot.executes p l xs (RealStore.rows p row) (RealStore.columns p column) ki
      (RealLoads.a_readback p s xs _ _ ki hinput hM)
      (RealLoads.b_readback p s xs _ _ ki hinput hN)
      (by simp [l,RealLoads.result,ha])
  have h3 := RealFinish.executes p (middle p s xs row column ki) row column ki
    (RealMath.blocks p.K p.BK) (values p xs row column ki)
    (by simp [middle,RealLoads.result,hk]) (by simp [middle,RealLoads.result,hcount])
    (by simp [middle,RealLoads.result,hr]) (by simp [middle,RealLoads.result,hc])
    (by simp [middle,RealDot.accumulator,values])
  rw [matches_source,code,List.append_assoc,stepStmts.append_some h1,stepStmts.append_some h2]
  exact h3

/-- Full raw-cell frame for one source iteration, including both completion
branches, masked lanes and stride gaps. -/
theorem frame (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (row column ki : Nat)
    (r : RegionName) (o : Nat)
    (h : ∀ i, RealStore.active p row column i → ¬(p.C=r ∧ RealStore.offset p row column i=o)) :
    (result p s xs row column ki).mem r o = s.mem r o := by
  unfold result
  split
  · exact RealStore.frame p (middle p s xs row column ki) row column (values p xs row column ki) r o h
  · rfl

/-- At the actual last-K store every active lane receives the independent full
matrix product; output injectivity is required only for active store lanes. -/
theorem complete_readback (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (row column ki : Nat)
    (hBK : 0 < p.BK) (hlast : ki+1=RealMath.blocks p.K p.BK)
    (i : TileIndex [p.BM,p.BN]) (hi : RealStore.active p row column i)
    (hinj : ∀ a b, RealStore.active p row column a → RealStore.active p row column b →
      RealStore.offset p row column a = RealStore.offset p row column b → a=b) :
    (result p s xs row column ki).mem p.C (RealStore.offset p row column i) =
      MemCell.real (RealMath.formula p.K xs (RealStore.rows p row i.1) (RealStore.columns p column i.2.1)) := by
  simp only [result,hlast,if_true]
  rw [RealStore.readback p (middle p s xs row column ki) row column (values p xs row column ki) i hi hinj]
  unfold values
  rw [hlast,RealTiles.completed_active p xs _ _ hBK hi.1 hi.2]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealStep
