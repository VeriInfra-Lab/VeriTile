import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatBlockPtr

/-! Typed block-pointer loads for the actual source's unmasked access. Axis
checks and both padding choices are retained. Only active reads require
allocated cells; unspecified inactive reads use an explicit oracle relation.
This file does not assert relocation for pointer-valued memory loads. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section

variable {α : Type} [Inhabited α]
variable (A : FlatAlloc) (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)
variable (s : State α)

/-- No in-bounds restriction: even an inactive address can alias another
region's encoding. This compatibility cannot be derived from allocation
disjointness alone. It allows any common value, including nonzero/NaN bits. -/
def UndefCompatible (d : TileDType) : Prop :=
  ∀ r o, flatUndef d A.flat (A.addr r o) = s.undef d r o

theorem undefCompatible_constant (d : TileDType) (value : Value α d)
    (hsource : ∀ r o, s.undef d r o = value)
    (hflat : ∀ r o, flatUndef d r o = value) :
    UndefCompatible A flatUndef s d := by
  intro r o
  rw [hsource, hflat]

/-- One lane of exactly the evaluator's block-pointer read. -/
def blockRead (p : Option ComputeDType) (d : TileDType) (padding : PaddingOption)
    (s : State α) (bp : BlockPtr) (ix checked : List Nat) : Value α d :=
  if bp.inBounds ix checked then
    (s.mem bp.region (bp.address ix)).read d
  else match padding with
    | .zero => boundaryZero M p d
    | .unspecified => s.undef d bp.region (bp.address ix)

theorem blockRead_flat (hd : A.Disjoint) (p : Option ComputeDType)
    (d : TileDType) (hp : d ≠ .ptr) (hb : d ≠ .blockPtr)
    (padding : PaddingOption) (bp : BlockPtr) (ix checked : List Nat)
    (hactive : bp.inBounds ix checked = Bool.true →
      bp.region ∈ A.regions ∧ bp.address ix < A.extent bp.region)
    (hu : UndefCompatible A flatUndef s d) :
    blockRead M p d padding (flattenState A background flatUndef s)
        (trValue A (α := α) .blockPtr bp) ix checked = blockRead M p d padding s bp ix checked := by
  unfold blockRead
  rw [trBlockPtr_inBounds]
  by_cases hi : bp.inBounds ix checked = Bool.true
  · simp only [hi, ite_true]
    rw [trBlockPtr_region, trBlockPtr_address,
      mem_addr A hd background flatUndef s (hactive hi).1 (hactive hi).2,
      trCell_read_data A _ d hp hb]
  · simp only [hi, Bool.false_eq_true, ite_false]
    cases padding with
    | zero => rfl
    | unspecified =>
        rw [trBlockPtr_region, trBlockPtr_address]
        change flatUndef d A.flat (A.addr bp.region (bp.address ix)) =
          s.undef d bp.region (bp.address ix)
        exact hu bp.region (bp.address ix)

theorem eval_load_blockPtr_none (p : Option ComputeDType) (d : TileDType)
    {sh : TileShape} (ptr : Op .blockPtr sh) (checked : List Nat) (padding : PaddingOption) :
    evalOp M p (.load d (.blockPtr ptr checked) .none padding) s =
      (evalOp M p ptr s).bind (fun ps => some (fun i =>
        blockRead M p d padding s (ps i) (TileShape.indexToList sh i) checked)) := rfl

/-- Full equality of successful values and failures. The hypotheses mention
only the immediate pointer child and the cells actually read by this load. -/
theorem op_load_blockPtr_none (hd : A.Disjoint) (p : Option ComputeDType)
    (d : TileDType) (hp : d ≠ .ptr) (hb : d ≠ .blockPtr)
    {sh : TileShape} (ptr : Op .blockPtr sh) (checked : List Nat) (padding : PaddingOption)
    (hptr : OpCommutes A M background flatUndef s p ptr)
    (hactive : ∀ ps, evalOp M p ptr s = some ps → ∀ i,
      (ps i).inBounds (TileShape.indexToList sh i) checked = Bool.true →
      (ps i).region ∈ A.regions ∧
        (ps i).address (TileShape.indexToList sh i) < A.extent (ps i).region)
    (hu : UndefCompatible A flatUndef s d) :
    OpSame A M background flatUndef s p (.load d (.blockPtr ptr checked) .none padding) := by
  unfold OpSame OpCommutes at *
  rw [FlatAlloc.flattenOp_load (padding := padding)]
  change evalOp M p (.load d (.blockPtr (A.flattenOp ptr) checked) .none padding)
      (flattenState A background flatUndef s) =
    evalOp M p (.load d (.blockPtr ptr checked) .none padding) s
  rw [eval_load_blockPtr_none, eval_load_blockPtr_none, hptr]
  cases hps : evalOp M p ptr s with
  | none => rfl
  | some ps =>
      simp only [Option.map_some, Option.bind_some]
      apply congrArg some
      funext i
      exact blockRead_flat A M background flatUndef s hd p d hp hb padding
        (ps i) (TileShape.indexToList sh i) checked (hactive ps hps i) hu

#axiomsClean undefCompatible_constant
#axiomsClean blockRead_flat
#axiomsClean eval_load_blockPtr_none
#axiomsClean op_load_blockPtr_none

end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
