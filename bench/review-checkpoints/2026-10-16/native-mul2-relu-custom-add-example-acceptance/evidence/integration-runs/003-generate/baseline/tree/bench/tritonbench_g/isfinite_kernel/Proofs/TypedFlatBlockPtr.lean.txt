import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatData

/-! Discrete block-pointer relocation and dynamic construction. Geometry is
proved for arbitrary rank. Dynamic offsets retain their exact evaluation
order and failure behavior. Only immediate child expression commutation is
used; no memory, whole-body, or whole-kernel execution premise appears. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section

variable {α : Type} [Inhabited α]
variable (A : FlatAlloc)

theorem trBlockPtr_region (bp : BlockPtr) :
    (trValue A (α := α) .blockPtr bp).region = A.flat := rfl

theorem trBlockPtr_address (bp : BlockPtr) (ix : List Nat) :
    (trValue A (α := α) .blockPtr bp).address ix = A.addr bp.region (bp.address ix) := by
  simp [trValue, BlockPtr.address, FlatAlloc.addr, Nat.add_assoc]

theorem trBlockPtr_inBounds (bp : BlockPtr) (ix checked : List Nat) :
    (trValue A (α := α) .blockPtr bp).inBounds ix checked = bp.inBounds ix checked := rfl

theorem trBlockPtr_metadata (bp : BlockPtr) :
    (trValue A (α := α) .blockPtr bp).parentShape = bp.parentShape ∧
    (trValue A (α := α) .blockPtr bp).blockShape = bp.blockShape ∧
    (trValue A (α := α) .blockPtr bp).strides = bp.strides ∧
    (trValue A (α := α) .blockPtr bp).offsets = bp.offsets :=
  ⟨rfl, rfl, rfl, rfl⟩

variable (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)
variable (s : State α) (p : Option ComputeDType)

theorem evalOffsets_commutes (offsets : List (Op .nat []))
    (he : ∀ e ∈ offsets, OpSame A M background flatUndef s p e) :
    evalOffsets M p (offsets.map A.flattenOp) (flattenState A background flatUndef s) =
      evalOffsets M p offsets s := by
  revert he
  induction offsets with
  | nil => intro he; rfl
  | cons e offsets ih =>
      intro he
      have head : evalOp M p (A.flattenOp e) (flattenState A background flatUndef s) =
          evalOp M p e s := he e (List.mem_cons_self ..)
      have tail := ih (fun e he' => he e (List.mem_cons_of_mem _ he'))
      change ((evalOp M p (A.flattenOp e) (flattenState A background flatUndef s)).bind fun v =>
        (evalOffsets M p (offsets.map A.flattenOp) (flattenState A background flatUndef s)).bind
          fun vs => some (v PUnit.unit :: vs)) =
        ((evalOp M p e s).bind fun v =>
          (evalOffsets M p offsets s).bind fun vs => some (v PUnit.unit :: vs))
      rw [head, tail]

/-- The additional base address is exactly the syntax inserted by flattenOp. -/
theorem eval_shifted_base (r : RegionName) (base : Op .nat [])
    (he : OpSame A M background flatUndef s p base) :
    evalOp M p (.add .nat .nil (.constNat (A.base r)) (A.flattenOp base))
        (flattenState A background flatUndef s) =
      (evalOp M p base s).map (fun v => bop (numeric M p .add .nat) .nil (fun _ => A.base r) v) := by
  change ((evalOp M p (A.flattenOp base) (flattenState A background flatUndef s)).bind
    fun v => some (bop (numeric M p .add .nat) .nil (fun _ => A.base r) v)) = _
  unfold OpSame at he
  rw [he]
  cases evalOp M p base s <;> rfl

theorem op_makeBlockPtrDynOffsets (r : RegionName) (base : Op .nat [])
    (parent block strides : List Nat) (offsets : List (Op .nat []))
    (hbase : OpSame A M background flatUndef s p base)
    (hoffsets : ∀ e ∈ offsets, OpSame A M background flatUndef s p e) :
    OpCommutes A M background flatUndef s p
      (.makeBlockPtrDynOffsets r base parent block strides offsets) := by
  have hattach : (offsets.attach.map fun e => A.flattenOp e.1) = offsets.map A.flattenOp := by
    simp
  have ho := evalOffsets_commutes A M background flatUndef s p offsets hoffsets
  have hb := eval_shifted_base A M background flatUndef s p r base hbase
  unfold OpCommutes
  rw [FlatAlloc.flattenOp]
  change ((evalOp M p (.add .nat .nil (.constNat (A.base r)) (A.flattenOp base))
      (flattenState A background flatUndef s)).bind fun b =>
    (evalOffsets M p (offsets.attach.map fun e => A.flattenOp e.1)
      (flattenState A background flatUndef s)).bind fun os =>
        some (fun (_ : TileIndex block) =>
          (⟨A.flat, b PUnit.unit, parent, block, strides, os⟩ : BlockPtr))) =
    (((evalOp M p base s).bind fun b => (evalOffsets M p offsets s).bind fun os =>
      some (fun (_ : TileIndex block) =>
        (⟨r, b PUnit.unit, parent, block, strides, os⟩ : BlockPtr))).map (trValues A (α := α) (d := .blockPtr) (sh := block)))
  rw [hattach, ho, hb]
  cases hbe : evalOp M p base s <;> cases hoe : evalOffsets M p offsets s <;>
    simp only [Option.map_none, Option.map_some, Option.bind_none, Option.bind_some]
  apply congrArg some
  funext i
  rfl

#axiomsClean trBlockPtr_region
#axiomsClean trBlockPtr_address
#axiomsClean trBlockPtr_inBounds
#axiomsClean trBlockPtr_metadata
#axiomsClean evalOffsets_commutes
#axiomsClean eval_shifted_base
#axiomsClean op_makeBlockPtrDynOffsets

end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
