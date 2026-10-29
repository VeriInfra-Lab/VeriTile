import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealProgram
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealDot
import VeriTile.Triton.KernelLemmas.ScatterStore

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealStore
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 14000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
local instance : Inhabited Stmt := ⟨.assign .nat [] "" (.constNat 0)⟩

def code (p : Parameters) : List Stmt := RealProgram.thenBody (RealProgram.body p)[8]!
def rows (p : Parameters) (row : Nat) : Fin p.BM → Nat := fun i => row*p.BM+i.val
def columns (p : Parameters) (column : Nat) : Fin p.BN → Nat := fun j => column*p.BN+j.val

def active (p : Parameters) (row column : Nat) (i : TileIndex [p.BM,p.BN]) : Prop :=
  rows p row i.1 < p.M ∧ columns p column i.2.1 < p.N
instance (p : Parameters) (row column : Nat) : DecidablePred (active p row column) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))
def offset (p : Parameters) (row column : Nat) (i : TileIndex [p.BM,p.BN]) : Nat :=
  p.scm * rows p row i.1 + p.scn * columns p column i.2.1

def prepared (p : Parameters) (s : BlockState) (row column : Nat)
    (values : Fin p.BM → Fin p.BN → ℝ) : BlockState :=
  let t := s.setReg "offs_cm" .int [p.BM] (Tile.vec fun i => (rows p row i : Int))
  let t := t.setReg "offs_cn" .int [p.BN] (Tile.vec fun j => (columns p column j : Int))
  let t := t.setReg "c_ptrs" .ptr [p.BM,p.BN] (Tile.mat fun i j =>
    (p.C, offset p row column (i,j,PUnit.unit)))
  let t := t.setReg "c_mask" .bool [p.BM,p.BN] (Tile.mat fun i j =>
    decide (active p row column (i,j,PUnit.unit)))
  t.setReg "c" .real [p.BM,p.BN] (Tile.mat fun i j => some (values i j))

def write (p : Parameters) (s : BlockState) (row column : Nat)
    (values : Fin p.BM → Fin p.BN → ℝ) : BlockState :=
  (TileShape.allIndices [p.BM,p.BN]).foldl (fun t i =>
    if active p row column i then t.writeMem p.C (offset p row column i) (values i.1 i.2.1)
    else t) s

def result (p : Parameters) (s : BlockState) (row column : Nat)
    (values : Fin p.BM → Fin p.BN → ℝ) : BlockState :=
  (write p (prepared p s row column values) row column values).setReg "accumulator"
    .real [p.BM,p.BN] (Tile.mat fun _ _ => some 0)

/-- The actual seven completion statements execute the masked source store and
reset the fp32-erased accumulator. There is no assumed output write. -/
theorem executes (p : Parameters) (s : BlockState) (row column : Nat)
    (values : Fin p.BM → Fin p.BN → ℝ)
    (hr : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hc : s.regs .int [] "pid_n" = some (Tile.scalar (column : Int)))
    (hv : s.regs .real [p.BM,p.BN] "accumulator" = some (Tile.mat fun i j => some (values i j))) :
    stepStmts (code p) s = some (result p s row column values) := by
  simp only [code, RealProgram.body, RealSource.kernel,
    Parameters.source, TypedSource.kernel, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, FloatDType.toTileDType,
    List.getElem!_cons_zero, List.getElem!_cons_succ, RealProgram.loopBody, RealProgram.thenBody]
  simp [prepared, result, write, rows, columns, offset, active,
    stepStmts, stepStmt, evalOp.eq_def, hr, hc, hv, bind, Option.bind,
    Tile.scalar, Tile.mat, Tile.vec, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    BlockState.writeMemTyped_real, -Tile.scalar_eta]
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast, Int.ofNat_lt]
  try rfl

/-- Every raw memory cell outside the active output addresses is preserved. -/
theorem frame (p : Parameters) (s : BlockState) (row column : Nat)
    (values : Fin p.BM → Fin p.BN → ℝ) (r : RegionName) (o : Nat)
    (h : ∀ i, active p row column i → ¬(p.C = r ∧ offset p row column i = o)) :
    (result p s row column values).mem r o = s.mem r o := by
  exact foldl_store_preserve_cell _ _ _ r o _ (prepared p s row column values)
    (fun i _ hi => h i hi)

/-- Active output lanes receive their values when the output strides identify
active lanes uniquely. No injectivity is required of masked lanes. -/
theorem readback (p : Parameters) (s : BlockState) (row column : Nat)
    (values : Fin p.BM → Fin p.BN → ℝ) (i : TileIndex [p.BM,p.BN])
    (hi : active p row column i)
    (hinj : ∀ a b, active p row column a → active p row column b →
      offset p row column a = offset p row column b → a = b) :
    (result p s row column values).mem p.C (offset p row column i) =
      MemCell.real (values i.1 i.2.1) := by
  obtain ⟨before,after,hsplit⟩ := List.append_of_mem (TileShape.mem_allIndices [p.BM,p.BN] i)
  have hn := TileShape.allIndices_nodup [p.BM,p.BN]
  rw [hsplit, List.nodup_append, List.nodup_cons] at hn
  have hafter : ∀ k ∈ after, active p row column k →
      ¬(p.C = p.C ∧ offset p row column k = offset p row column i) := by
    intro k hk ha he
    have heq := hinj k i ha hi he.2
    subst k
    exact hn.2.1.1 hk
  change (write p (prepared p s row column values) row column values).mem _ _ = _
  unfold write
  rw [hsplit, List.foldl_append, List.foldl_cons, if_pos hi,
    foldl_store_preserve_cell _ _ _ _ _ after _ hafter,
    BlockState.writeMem_mem, if_pos ⟨rfl,rfl⟩]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealStore
