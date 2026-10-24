import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatExpr
import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatBlockPtr
import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatScatter

/-! The source's Bool block-pointer store. Both executions retain exactly
`TileShape.allIndices` order; aliasing lanes are neither sorted nor removed.
Only active cells require allocation bounds. This store needs no unspecified
value compatibility, and does not reinterpret Bool output through Real. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section

variable {α : Type} [Inhabited α]
variable (A : FlatAlloc) (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)
variable (s : State α)

def blockStoreBool {sh : TileShape} (checked : List Nat)
    (ps : Values α .blockPtr sh) (values : Values α .bool sh) (s : State α) : State α :=
  (TileShape.allIndices sh).foldl (fun t i =>
    if (ps i).inBounds (TileShape.indexToList sh i) checked then
      t.write (ps i).region ((ps i).address (TileShape.indexToList sh i)) (.mk .bool (values i))
    else t) s

theorem blockStoreBool_flat (hd : A.Disjoint) {sh : TileShape} (checked : List Nat)
    (ps : Values α .blockPtr sh) (values : Values α .bool sh)
    (hactive : ∀ i, (ps i).inBounds (TileShape.indexToList sh i) checked = Bool.true →
      (ps i).region ∈ A.regions ∧
        (ps i).address (TileShape.indexToList sh i) < A.extent (ps i).region) :
    blockStoreBool checked (trValues A ps) values (flattenState A background flatUndef s) =
      flattenState A background flatUndef (blockStoreBool checked ps values s) := by
  have h := flatten_masked_tile A hd background flatUndef
    (fun i => (ps i).region)
    (fun i => (ps i).address (TileShape.indexToList sh i))
    (fun i => Cell.mk .bool (values i))
    (fun i => (ps i).inBounds (TileShape.indexToList sh i) checked) s hactive
  unfold blockStoreBool
  simp only [trValues, trBlockPtr_region, trBlockPtr_address, trBlockPtr_inBounds]
  exact h.symm

theorem store_blockPtr_bool_none {sh : TileShape} (ptr : Op .blockPtr sh)
    (checked : List Nat) (values : Values α .bool sh) :
    FP.Structural.store M (.blockPtr ptr checked) .none values s =
      (evalOp M none ptr s).map (fun ps => blockStoreBool checked ps values s) := by
  cases hp : evalOp M none ptr s <;>
    simp only [FP.Structural.store, hp, bind, Option.bind, Option.pure_def, Option.map]
  rfl

theorem store_bool_none_commutes (hd : A.Disjoint) {sh : TileShape}
    (ptr : Op .blockPtr sh) (checked : List Nat) (values : Values α .bool sh)
    (hptr : OpCommutes A M background flatUndef s none ptr)
    (hactive : ∀ ps, evalOp M none ptr s = some ps → ∀ i,
      (ps i).inBounds (TileShape.indexToList sh i) checked = Bool.true →
      (ps i).region ∈ A.regions ∧
        (ps i).address (TileShape.indexToList sh i) < A.extent (ps i).region) :
    FP.Structural.store M (.blockPtr (A.flattenOp ptr) checked) .none values
        (flattenState A background flatUndef s) =
      (FP.Structural.store M (.blockPtr ptr checked) .none values s).map
        (flattenState A background flatUndef) := by
  unfold OpCommutes at hptr
  rw [store_blockPtr_bool_none, store_blockPtr_bool_none, hptr]
  cases hp : evalOp M none ptr s with
  | none => rfl
  | some ps =>
      simp only [Option.map_some]
      apply congrArg some
      exact blockStoreBool_flat A background flatUndef s hd checked ps values (hactive ps hp)

/-- Immediate source statement transport, preserving value failure, pointer
failure and complete successful states. Its children are evaluated before
the ordered writes in both runs, exactly as Structural.step/store require. -/
theorem step_store_bool_none (hd : A.Disjoint) {sh : TileShape}
    (ptr : Op .blockPtr sh) (checked : List Nat) (value : ComputeExpr .bool sh)
    (hvalue : ExprSame A M background flatUndef s value)
    (hptr : OpCommutes A M background flatUndef s none ptr)
    (hactive : ∀ ps, evalOp M none ptr s = some ps → ∀ i,
      (ps i).inBounds (TileShape.indexToList sh i) checked = Bool.true →
      (ps i).region ∈ A.regions ∧
        (ps i).address (TileShape.indexToList sh i) < A.extent (ps i).region) :
    FP.Structural.step M
        (flattenComputeStmt A (.store .bool sh (.blockPtr ptr checked) value .none))
        (flattenState A background flatUndef s) =
      (FP.Structural.step M (.store .bool sh (.blockPtr ptr checked) value .none) s).map
        (flattenState A background flatUndef) := by
  unfold ExprSame at hvalue
  rw [flattenComputeStmt, FP.Structural.step, FP.Structural.step]
  change ((evalExpr M (flattenExpr A value) (flattenState A background flatUndef s)).bind
    fun values => FP.Structural.store M (.blockPtr (A.flattenOp ptr) checked) .none values
      (flattenState A background flatUndef s)) =
    (((evalExpr M value s).bind fun values =>
      FP.Structural.store M (.blockPtr ptr checked) .none values s).map
        (flattenState A background flatUndef))
  rw [hvalue]
  cases hv : evalExpr M value s with
  | none => rfl
  | some values =>
      simp only [Option.bind_some]
      exact store_bool_none_commutes A M background flatUndef s hd ptr checked values hptr hactive

#axiomsClean blockStoreBool_flat
#axiomsClean store_blockPtr_bool_none
#axiomsClean store_bool_none_commutes
#axiomsClean step_store_bool_none

end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
