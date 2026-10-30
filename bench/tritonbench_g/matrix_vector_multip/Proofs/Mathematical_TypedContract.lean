import bench.tritonbench_g.matrix_vector_multip.Proofs.Mathematical_TypedExecution
namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def rowIndex {α : Type} (p : Parameters) (s : State α) (i : Fin p.blockRows) :=
  s.pids 0 * p.blockRows + i.val

def inputValue {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (s : State α) (r : RegionName) (off : Nat) (active : Prop) [Decidable active] : α :=
  M.cast (some .fp32) d .real (if active then toFloat d ((s.mem r off).read d.toTileDType)
    else M.cast (some .fp32) .real d (M.literal (some .fp32) .real 0))

/-- The original accumulator starts at +0 and adds a separately rounded
product. That add is retained even for a single iteration. -/
def accumulated {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (i : TileIndex [p.blockRows,p.blockColumns]) : α :=
  let zero := M.literal (some .fp32) .real 0
  if 0 < p.columns then
    M.binary (some .fp32) .real .add zero
      (M.binary (some .fp32) .real .mul
        (inputValue M d s p.a (rowIndex p s i.1 * p.strideRow + i.2.1.val * p.strideColumn)
          (rowIndex p s i.1 < p.rows ∧ i.2.1.val < p.columns))
        (inputValue M d s p.b (i.2.1.val * p.strideVector) (i.2.1.val < p.columns)))
  else zero

def outputValues {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (i : TileIndex [p.blockRows,1]) : α :=
  M.cast none .real d (M.reduceSum (some .fp32) ⟨1, by simp⟩ Bool.false
    (accumulated M d p s) (i.1,PUnit.unit))

def outputOffset {α : Type} (p : Parameters) (s : State α)
    (i : TileIndex [p.blockRows,1]) : Nat := rowIndex p s i.1 * p.strideOut

def reference {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.blockRows,1]).foldl (fun t i =>
    if rowIndex p s i.1 < p.rows then t.write p.out (outputOffset p s i)
      (.mk d.toTileDType (ofFloat d (outputValues M d p s i))) else t) s

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (hw : 0 < p.blockColumns) (hcols : p.columns ≤ p.blockColumns) :
    (execution M d p s hw hcols).val.mem = (reference M d p s).mem := by
  rcases p with ⟨a,b,o,n,m,ar,ac,bc,oc,bn,bm⟩
  by_cases hm : 0 < m
  all_goals cases d
  all_goals simp [execution, reference, outputValues, accumulated, inputValue,
    outputOffset, rowIndex, fold_writes_mem, State.setReg, ofFloat, toFloat,
    Region.cast, FloatDType.toTileDType, FloatDType.numericDType, bop,
    Function.comp_def, Broadcast.leftIndex, Broadcast.rightIndex, hm]
  all_goals try unfold accumulated
  all_goals simp [hm, inputValue, rowIndex, ofFloat, toFloat]
  all_goals norm_num only
  all_goals rfl
#axiomsClean execution_memory
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
