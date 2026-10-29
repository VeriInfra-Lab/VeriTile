import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatHeader
import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatLoad
import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatStore

/-! Exact source-body expressions and their immediate transport obligations.
The body equations connect these names to both actual quoted sources; no
statement, precision, cast, boundary check, or padding choice is removed. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section

def sourcePtr (p : Parametric.Parameters) (r : RegionName) (stride : Nat) :
    Op .blockPtr [p.block] :=
  .makeBlockPtrDynOffsets r (.constNat 0) [p.size] [p.block] [stride]
    [.ref .nat [] "offset0"]

def sourceLoad (p : Parametric.Parameters) (wide : Bool) : ComputeExpr .real [p.block] :=
  if wide then .compute (.alg .fp64 (.castFloat .real .real
    (.load .real (.blockPtr (.ref .blockPtr [p.block] "in0_bptr") [0]) .none .unspecified)))
  else .compute (.alg .fp32 (.castFloat .real .real
    (.load .real (.blockPtr (.ref .blockPtr [p.block] "in0_bptr") [0]) .none .unspecified)))

def sourceClassify (p : Parametric.Parameters) (wide : Bool) : ComputeExpr .bool [p.block] :=
  if wide then .isFinite .fp64 (.compute (.alg .fp64 (.ref .real [p.block] "in0")))
  else .isFinite .fp32 (.compute (.alg .fp32
    (.castFloat .real .real (.ref .real [p.block] "in0"))))

def sourceBody (p : Parametric.Parameters) (wide : Bool) (tile : Op .nat []) :
    List ComputeStmt :=
  [.assign .nat [] "tile_id" (.alg tile),
   .assign .nat [] "tile_id0" (.alg (.ref .nat [] "tile_id")),
   .assign .nat [] "offset0" (.alg
     (.mul .nat .nil (.ref .nat [] "tile_id0") (.constNat p.block))),
   .assign .blockPtr [p.block] "in0_bptr" (.alg (sourcePtr p p.input p.inputStride)),
   .assign .real [p.block] "in0" (sourceLoad p wide),
   .assign .bool [p.block] "out0" (sourceClassify p wide),
   .assign .blockPtr [p.block] "out0_bptr" (.alg (sourcePtr p p.output p.outputStride)),
   .store .bool [p.block] (.blockPtr (.ref .blockPtr [p.block] "out0_bptr") [0])
     (.alg (.ref .bool [p.block] "out0")) .none]

def sourceLoopTile : Op .nat [] :=
  .add .nat .nil (.ref .nat [] "pid")
    (.mul .nat .nil (.ref .nat [] "j") (.ref .nat [] "num_ctas"))

theorem source_one32_exact (p : Parametric.Parameters) :
    Parametric.oneBody32 p = sourceBody p Bool.false (.ref .nat [] "pid") := rfl
theorem source_one64_exact (p : Parametric.Parameters) :
    Parametric.oneBody64 p = sourceBody p Bool.true (.ref .nat [] "pid") := rfl
theorem source_loop32_exact (p : Parametric.Parameters) :
    Parametric.loopBody32 p = sourceBody p Bool.false sourceLoopTile := rfl
theorem source_loop64_exact (p : Parametric.Parameters) :
    Parametric.loopBody64 p = sourceBody p Bool.true sourceLoopTile := rfl

variable {α : Type} [Inhabited α]
variable (A : FlatAlloc) (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)

/-- Only successfully obtained pointer lanes which pass the actual boundary
check must be allocated. Arbitrary initial stale registers are allowed. -/
def SourcePtrSafe (p : Parametric.Parameters) (name : RegName) (s : State α) : Prop :=
  ∀ ps, s.regs .blockPtr [p.block] name = some ps → ∀ i,
    (ps i).inBounds (TileShape.indexToList [p.block] i) [0] = Bool.true →
    (ps i).region ∈ A.regions ∧
      (ps i).address (TileShape.indexToList [p.block] i) < A.extent (ps i).region

theorem source_ptr_commutes (p : Parametric.Parameters) (r : RegionName) (stride : Nat)
    (s : State α) : OpCommutes A M background flatUndef s none (sourcePtr p r stride) := by
  apply op_makeBlockPtrDynOffsets A M background flatUndef s none
  · exact op_constNat A M background flatUndef s none 0
  · intro e he
    simp only [List.mem_singleton] at he
    subst e
    exact op_ref_data A M background flatUndef s none .nat [] "offset0" (by decide) (by decide)

theorem source_load_same (hd : A.Disjoint) (p : Parametric.Parameters) (wide : Bool)
    (s : State α) (hs : SourcePtrSafe A p "in0_bptr" s)
    (hu : UndefCompatible A flatUndef s .real) :
    ExprSame A M background flatUndef s (sourceLoad p wide) := by
  have hload (prec : Option ComputeDType) : OpSame A M background flatUndef s prec
      (.load .real (.blockPtr (.ref .blockPtr [p.block] "in0_bptr") [0]) .none .unspecified) := by
    apply op_load_blockPtr_none A M background flatUndef s hd prec .real
      (by decide) (by decide)
    · exact op_ref_commutes A M background flatUndef s prec .blockPtr [p.block] "in0_bptr"
    · intro ps hp
      exact hs ps hp
    · exact hu
  cases wide <;> unfold sourceLoad <;> simp only [Bool.false_eq_true, if_false, if_true]
  · apply expr_compute A M background flatUndef s
    apply compute_alg A M background flatUndef s
    exact op_castFloat A M background flatUndef s (some .fp32) .real .real _ (hload _)
  · apply expr_compute A M background flatUndef s
    apply compute_alg A M background flatUndef s
    exact op_castFloat A M background flatUndef s (some .fp64) .real .real _ (hload _)

theorem source_classify_same (p : Parametric.Parameters) (wide : Bool) (s : State α) :
    ExprSame A M background flatUndef s (sourceClassify p wide) := by
  cases wide <;> unfold sourceClassify <;> simp only [Bool.false_eq_true, if_false, if_true]
  · apply expr_isFinite A M background flatUndef s
    apply expr_compute A M background flatUndef s
    apply compute_alg A M background flatUndef s
    apply op_castFloat A M background flatUndef s
    exact op_ref_data A M background flatUndef s (some .fp32) .real [p.block] "in0"
      (by decide) (by decide)
  · apply expr_isFinite A M background flatUndef s
    apply expr_compute A M background flatUndef s
    apply compute_alg A M background flatUndef s
    exact op_ref_data A M background flatUndef s (some .fp64) .real [p.block] "in0"
      (by decide) (by decide)

theorem assign_result (d : AlgDType) (sh : TileShape) (name : RegName)
    (e : ComputeExpr d sh) (s t : State α)
    (ht : FP.Structural.step M (.assign d sh name e) s = some t) :
    ∃ v, evalExpr M e s = some v ∧ t = s.setReg name d sh v := by
  rw [FP.Structural.step] at ht
  cases he : evalExpr M e s with
  | none => simp only [he, bind, Option.bind_none] at ht; cases ht
  | some v =>
      simp only [he, bind, Option.bind_some, Option.pure_def, Option.some.injEq] at ht
      exact ⟨v, rfl, ht.symm⟩

theorem assign_undef (d : AlgDType) (sh : TileShape) (name : RegName)
    (e : ComputeExpr d sh) (s t : State α)
    (ht : FP.Structural.step M (.assign d sh name e) s = some t) : t.undef = s.undef := by
  obtain ⟨v, _, rfl⟩ := assign_result M d sh name e s t ht
  rfl

theorem assign_compatible (d : AlgDType) (sh : TileShape) (name : RegName)
    (e : ComputeExpr d sh) (s t : State α)
    (ht : FP.Structural.step M (.assign d sh name e) s = some t)
    (hu : UndefCompatible A flatUndef s .real) : UndefCompatible A flatUndef t .real := by
  unfold UndefCompatible at *
  rw [assign_undef M d sh name e s t ht]
  exact hu

/-- The actual source constructor establishes bounded access independently
of the offset value, including offsets beyond the logical tensor. -/
theorem source_ptr_safe_after (p : Parametric.Parameters) (r : RegionName) (stride : Nat)
    (name : RegName) (s t : State α)
    (hr : r ∈ A.regions) (hb : ∀ x, x < p.size → x * stride < A.extent r)
    (ht : FP.Structural.step M (.assign .blockPtr [p.block] name
      (.alg (sourcePtr p r stride))) s = some t) : SourcePtrSafe A p name t := by
  obtain ⟨v, hv, rfl⟩ := assign_result M .blockPtr [p.block] name
    (.alg (sourcePtr p r stride)) s t ht
  cases ho : s.regs .nat [] "offset0" with
  | none =>
      simp only [sourcePtr, evalExpr, evalOp_unfold, evalOffsets, ho,
        bind, Option.bind, Option.pure_def] at hv
      cases hv
  | some offsets =>
      have he : v = fun (_ : TileIndex [p.block]) =>
          (⟨r, 0, [p.size], [p.block], [stride], [offsets PUnit.unit]⟩ : BlockPtr) := by
        simpa only [sourcePtr, evalExpr, evalOp_unfold, evalOffsets, ho,
          bind, Option.bind, Option.pure_def, Option.some.injEq] using hv.symm
      subst v
      intro ps hp i hi
      rw [State.setReg_same] at hp
      have hps := Option.some.inj hp
      subst ps
      simp only [TileShape.indexToList, BlockPtr.inBounds_1d, decide_eq_true_eq] at hi
      refine ⟨hr, ?_⟩
      simpa only [TileShape.indexToList, BlockPtr.address_1d, Nat.zero_add] using
        hb (offsets PUnit.unit + i.1.val) hi

#axiomsClean source_one32_exact
#axiomsClean source_one64_exact
#axiomsClean source_loop32_exact
#axiomsClean source_loop64_exact
#axiomsClean source_ptr_commutes
#axiomsClean source_load_same
#axiomsClean source_classify_same
#axiomsClean assign_result
#axiomsClean assign_undef
#axiomsClean assign_compatible
#axiomsClean source_ptr_safe_after
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
