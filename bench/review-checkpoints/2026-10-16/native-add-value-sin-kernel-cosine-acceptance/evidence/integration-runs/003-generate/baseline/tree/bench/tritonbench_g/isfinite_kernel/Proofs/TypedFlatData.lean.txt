import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatState

/-! Pure typed expression relocation. `OpSame` is the data-channel interface;
`OpCommutes` also expresses translated pointer payloads. Each constructor
lemma consumes only its children's equalities. There is no source/kernel
equivalence premise and no numerical law on the floating algebra. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section

variable {α : Type} [Inhabited α]
variable (A : FlatAlloc) (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)
variable (s : State α)

def OpSame (p : Option ComputeDType) {d : TileDType} {sh : TileShape}
    (e : Op d sh) : Prop :=
  evalOp M p (A.flattenOp e) (flattenState A background flatUndef s) = evalOp M p e s

def OpCommutes (p : Option ComputeDType) {d : TileDType} {sh : TileShape}
    (e : Op d sh) : Prop :=
  evalOp M p (A.flattenOp e) (flattenState A background flatUndef s) =
    (evalOp M p e s).map (trValues A)

theorem trValues_data {d : TileDType} {sh : TileShape} (v : Values α d sh)
    (hp : d ≠ .ptr) (hb : d ≠ .blockPtr) : trValues A v = v := by
  funext i
  cases d <;> simp_all [trValues, trValue]

theorem map_trValues_data {d : TileDType} {sh : TileShape}
    (v : Option (Values α d sh)) (hp : d ≠ .ptr) (hb : d ≠ .blockPtr) :
    v.map (trValues A) = v := by
  cases v with
  | none => rfl
  | some v => rw [Option.map_some, trValues_data A v hp hb]

theorem opSame_iff_commutes (p : Option ComputeDType) {d : TileDType} {sh : TileShape}
    (e : Op d sh) (hp : d ≠ .ptr) (hb : d ≠ .blockPtr) :
    OpSame A M background flatUndef s p e ↔ OpCommutes A M background flatUndef s p e := by
  unfold OpSame OpCommutes
  rw [map_trValues_data A _ hp hb]

variable (p : Option ComputeDType)

/-- All register channels relocate, including missing and wrong-shape lookups. -/
theorem op_ref_commutes (d : TileDType) (sh : TileShape) (n : RegName) :
    OpCommutes A M background flatUndef s p (.ref d sh n) := by
  simp only [OpCommutes, FlatAlloc.flattenOp, evalOp_unfold, flattenState]

theorem op_ref_data (d : TileDType) (sh : TileShape) (n : RegName)
    (hp : d ≠ .ptr) (hb : d ≠ .blockPtr) :
    OpSame A M background flatUndef s p (.ref d sh n) := by
  apply (opSame_iff_commutes A M background flatUndef s p _ hp hb).2
  exact op_ref_commutes A M background flatUndef s p d sh n

theorem op_constNat (n : Nat) :
    OpSame A M background flatUndef s p (.constNat n) := by
  simp only [OpSame, FlatAlloc.flattenOp, evalOp_unfold]

theorem op_constInt (n : Int) :
    OpSame A M background flatUndef s p (.constInt n) := by
  simp only [OpSame, FlatAlloc.flattenOp, evalOp_unfold]

theorem op_constBool (b : Bool) :
    OpSame A M background flatUndef s p (.constBool b) := by
  simp only [OpSame, FlatAlloc.flattenOp, evalOp_unfold]

theorem op_const (x : ℝ) :
    OpSame A M background flatUndef s p (.const x) := by
  simp only [OpSame, FlatAlloc.flattenOp, evalOp_unfold]

theorem op_constFloat (d : FloatDType) (x : ℝ) :
    OpSame A M background flatUndef s p (.constFloat d x) := by
  simp only [OpSame, FlatAlloc.flattenOp, evalOp_unfold]

theorem op_programId (dim : Nat) :
    OpSame A M background flatUndef s p (.programId dim) := by
  simp only [OpSame, FlatAlloc.flattenOp, evalOp_unfold, flattenState]

theorem op_numPrograms (dim : Nat) :
    OpSame A M background flatUndef s p (.numPrograms dim) := by
  simp only [OpSame, FlatAlloc.flattenOp, evalOp_unfold, flattenState]

theorem op_arange (n : Nat) :
    OpSame A M background flatUndef s p (.arange n) := by
  simp only [OpSame, FlatAlloc.flattenOp, evalOp_unfold]

/-- The generic numeric constructor includes exact Nat arithmetic. Its
floating instances only reuse the same callback, without algebraic rewrites. -/
theorem op_add {d : TileDType} {a b out : TileShape} (nd : NumericDType d)
    (bc : Broadcast a b out) (x : Op d a) (y : Op d b)
    (hx : OpSame A M background flatUndef s p x)
    (hy : OpSame A M background flatUndef s p y) :
    OpSame A M background flatUndef s p (.add nd bc x y) := by
  unfold OpSame at *
  rw [FlatAlloc.flattenOp]
  change ((evalOp M p (A.flattenOp x) (flattenState A background flatUndef s)).bind fun vx =>
    (evalOp M p (A.flattenOp y) (flattenState A background flatUndef s)).bind fun vy =>
      some (bop (numeric M p .add nd) bc vx vy)) =
    ((evalOp M p x s).bind fun vx => (evalOp M p y s).bind fun vy =>
      some (bop (numeric M p .add nd) bc vx vy))
  rw [hx, hy]

theorem op_sub {d : TileDType} {a b out : TileShape} (nd : NumericDType d)
    (bc : Broadcast a b out) (x : Op d a) (y : Op d b)
    (hx : OpSame A M background flatUndef s p x)
    (hy : OpSame A M background flatUndef s p y) :
    OpSame A M background flatUndef s p (.sub nd bc x y) := by
  unfold OpSame at *
  rw [FlatAlloc.flattenOp]
  change ((evalOp M p (A.flattenOp x) (flattenState A background flatUndef s)).bind fun vx =>
    (evalOp M p (A.flattenOp y) (flattenState A background flatUndef s)).bind fun vy =>
      some (bop (numeric M p .sub nd) bc vx vy)) =
    ((evalOp M p x s).bind fun vx => (evalOp M p y s).bind fun vy =>
      some (bop (numeric M p .sub nd) bc vx vy))
  rw [hx, hy]

theorem op_mul {d : TileDType} {a b out : TileShape} (nd : NumericDType d)
    (bc : Broadcast a b out) (x : Op d a) (y : Op d b)
    (hx : OpSame A M background flatUndef s p x)
    (hy : OpSame A M background flatUndef s p y) :
    OpSame A M background flatUndef s p (.mul nd bc x y) := by
  unfold OpSame at *
  rw [FlatAlloc.flattenOp]
  change ((evalOp M p (A.flattenOp x) (flattenState A background flatUndef s)).bind fun vx =>
    (evalOp M p (A.flattenOp y) (flattenState A background flatUndef s)).bind fun vy =>
      some (bop (numeric M p .mul nd) bc vx vy)) =
    ((evalOp M p x s).bind fun vx => (evalOp M p y s).bind fun vy =>
      some (bop (numeric M p .mul nd) bc vx vy))
  rw [hx, hy]

theorem op_div {d : TileDType} {a b out : TileShape} (nd : NumericDType d)
    (bc : Broadcast a b out) (x : Op d a) (y : Op d b)
    (hx : OpSame A M background flatUndef s p x)
    (hy : OpSame A M background flatUndef s p y) :
    OpSame A M background flatUndef s p (.div nd bc x y) := by
  unfold OpSame at *
  rw [FlatAlloc.flattenOp]
  change ((evalOp M p (A.flattenOp x) (flattenState A background flatUndef s)).bind fun vx =>
    (evalOp M p (A.flattenOp y) (flattenState A background flatUndef s)).bind fun vy =>
      some (bop (numeric M p .div nd) bc vx vy)) =
    ((evalOp M p x s).bind fun vx => (evalOp M p y s).bind fun vy =>
      some (bop (numeric M p .div nd) bc vx vy))
  rw [hx, hy]

theorem op_full {d : TileDType} (sh : TileShape) (e : Op d [])
    (he : OpSame A M background flatUndef s p e) :
    OpSame A M background flatUndef s p (.full sh e) := by
  unfold OpSame at *
  rw [FlatAlloc.flattenOp]
  change ((evalOp M p (A.flattenOp e) (flattenState A background flatUndef s)).bind fun v =>
    some (fun (_ : TileIndex sh) => v PUnit.unit)) =
    ((evalOp M p e s).bind fun v => some (fun (_ : TileIndex sh) => v PUnit.unit))
  rw [he]

theorem op_broadcast {d : TileDType} (sh : TileShape) (e : Op d [])
    (he : OpSame A M background flatUndef s p e) :
    OpSame A M background flatUndef s p (.broadcast e sh) := by
  unfold OpSame at *
  rw [FlatAlloc.flattenOp]
  change ((evalOp M p (A.flattenOp e) (flattenState A background flatUndef s)).bind fun v =>
    some (fun (_ : TileIndex sh) => v PUnit.unit)) =
    ((evalOp M p e s).bind fun v => some (fun (_ : TileIndex sh) => v PUnit.unit))
  rw [he]

theorem op_castFloat {sh : TileShape} (src dst : FloatDType) (e : Op src.toTileDType sh)
    (he : OpSame A M background flatUndef s p e) :
    OpSame A M background flatUndef s p (.castFloat src dst e) := by
  unfold OpSame at *
  rw [FlatAlloc.flattenOp]
  change ((evalOp M p (A.flattenOp e) (flattenState A background flatUndef s)).bind fun v =>
    some (fun i => ofFloat dst (M.cast p src dst (toFloat src (v i))))) =
    ((evalOp M p e s).bind fun v =>
      some (fun i => ofFloat dst (M.cast p src dst (toFloat src (v i)))))
  rw [he]

/-- Child evaluation uses the source's explicit precision, not the ambient p. -/
theorem op_withPrecision {sh : TileShape} (d : FloatDType)
    (precision : IndexComputePrecision) (e : Op d.toTileDType sh)
    (he : OpSame A M background flatUndef s
      (some (match precision with | .fp32 => .fp32 | .fp64 => .fp64)) e) :
    OpSame A M background flatUndef s p (.withPrecision d precision e) := by
  unfold OpSame at *
  rw [FlatAlloc.flattenOp]
  conv_lhs => rw [evalOp_unfold]
  conv_rhs => rw [evalOp_unfold]
  exact he

theorem op_castNatToInt {sh : TileShape} (e : Op .nat sh)
    (he : OpSame A M background flatUndef s p e) :
    OpSame A M background flatUndef s p (.castNatToInt e) := by
  unfold OpSame at *
  rw [FlatAlloc.flattenOp]
  change ((evalOp M p (A.flattenOp e) (flattenState A background flatUndef s)).bind fun v =>
    some (fun i => Int.ofNat (v i))) =
    ((evalOp M p e s).bind fun v => some (fun i => Int.ofNat (v i)))
  rw [he]

theorem op_castIntToNat {sh : TileShape} (e : Op .int sh)
    (he : OpSame A M background flatUndef s p e) :
    OpSame A M background flatUndef s p (.castIntToNat e) := by
  unfold OpSame at *
  rw [FlatAlloc.flattenOp]
  change ((evalOp M p (A.flattenOp e) (flattenState A background flatUndef s)).bind fun v =>
    some (fun i => (v i).toNat)) =
    ((evalOp M p e s).bind fun v => some (fun i => (v i).toNat))
  rw [he]

theorem op_natToReal {sh : TileShape} (e : Op .nat sh)
    (he : OpSame A M background flatUndef s p e) :
    OpSame A M background flatUndef s p (.natToReal e) := by
  unfold OpSame at *
  rw [FlatAlloc.flattenOp]
  change ((evalOp M p (A.flattenOp e) (flattenState A background flatUndef s)).bind fun v =>
    some (fun i => M.fromNat p (v i))) =
    ((evalOp M p e s).bind fun v => some (fun i => M.fromNat p (v i)))
  rw [he]

#axiomsClean trValues_data
#axiomsClean map_trValues_data
#axiomsClean opSame_iff_commutes
#axiomsClean op_ref_commutes
#axiomsClean op_ref_data
#axiomsClean op_constNat
#axiomsClean op_constInt
#axiomsClean op_constBool
#axiomsClean op_const
#axiomsClean op_constFloat
#axiomsClean op_programId
#axiomsClean op_numPrograms
#axiomsClean op_arange
#axiomsClean op_add
#axiomsClean op_sub
#axiomsClean op_mul
#axiomsClean op_div
#axiomsClean op_full
#axiomsClean op_broadcast
#axiomsClean op_castFloat
#axiomsClean op_withPrecision
#axiomsClean op_castNatToInt
#axiomsClean op_castIntToNat
#axiomsClean op_natToReal

end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
