import bench.tritonbench_g.vector_addition.Proofs.Mathematical_HostGrid
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_DTypeScope
import VeriTile.Triton.Float.FixedInteger

/-! Explicit fixed-width integer/bool materialization of the original source.
A boolean pointer loads an int8 storage cell and compares it with zero. Its
compute value is unsigned i1, while stores convert to signed int8 storage.
Each promotion, modular addition and destination conversion is explicit.
The logical memory model still uses Int/Nat tagged cells; physical bytes,
source representability, overflow sanitizers and native execution are not
silently inferred from this fixed-width value graph.
-/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.IntegerSource
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters offset)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Format where
  width : Nat
  signed : Bool
  deriving DecidableEq, Repr

inductive Storage where
  | boolean | signed (width : Nat) | unsigned (width : Nat)
  deriving DecidableEq, Repr

def Storage.compute : Storage → Format
  | .boolean => ⟨1, Bool.false⟩
  | .signed width => ⟨width, Bool.true⟩
  | .unsigned width => ⟨width, Bool.false⟩

def Storage.storeFormat : Storage → Format
  | .boolean => ⟨8, Bool.true⟩
  | d => d.compute

def promote (x y : Storage) : Format :=
  let a := x.compute
  let b := y.compute
  if a.signed = b.signed then if b.width ≤ a.width then a else b
  else if a.signed then if b.width < a.width then a else b
  else if a.width < b.width then b else a

/-- Unsigned low-bit interpretation or signed two's-complement interpretation. -/
def castValue (f : Format) (v : Int) : Int :=
  if f.signed then (BitVec.ofInt f.width v).toInt
  else ((BitVec.ofInt f.width v).toNat : Int)

/-- Reuse the shared, proved fixed-width cast expressions. -/
def castOp (f : Format) (e : Op .int shape) : Op .int shape :=
  if f.signed then .unsignedToInt f.width (.intToUnsigned f.width e)
  else .castNatToInt (.intToUnsigned f.width e)

theorem eval_castOp {α : Type} [Inhabited α] (M : Algebra α)
    (f : Format) (e : Op .int shape) (s : State α) :
    FP.Structural.evalOp M none (castOp f e) s = (do
      let v ← FP.Structural.evalOp M none e s
      pure (fun i => castValue f (v i))) := by
  unfold castOp castValue
  split
  · rw [FP.Structural.eval_unsignedToInt, FP.Structural.eval_intToUnsigned]
    cases h : FP.Structural.evalOp M none e s <;> simp only [h, bind, pure, Option.bind_none, Option.bind_some]
    congr 1
    funext i
    rw [signedBits_eq_bitvec, unsigned_cast_eq_bitvec, BitVec.ofNat_toNat, BitVec.setWidth_eq]
  · simp only [FP.Structural.evalOp, FP.Structural.eval_intToUnsigned]
    cases h : FP.Structural.evalOp M none e s <;> simp only [h, bind, pure, Option.bind_none, Option.bind_some]
    congr 1
    funext i
    simp only [bop, integralMod, IntegerExpr.same_left, IntegerExpr.same_right]
    rw [unsigned_cast_eq_bitvec]
    rfl

/-- Bool pointer reinterpretation is represented by .int storage and !=0;
unsigned loads become nonnegative Int only for common subsequent casts. -/
def loadOp (d : Storage) (region : RegionName) (off : Op .nat shape)
    (mask : Op .bool shape) : Op .int shape :=
  match d with
  | .boolean =>
      .where (.ne .int (IntegerExpr.sameShape shape)
        (.load .int (.region region off) (.mask mask))
        (.full shape (.constInt 0)))
        (.full shape (.constInt 1)) (.full shape (.constInt 0))
  | .signed _ => .load .int (.region region off) (.mask mask)
  | .unsigned _ => .castNatToInt (.load .nat (.region region off) (.mask mask))

def loaded {α : Type} [Inhabited α] (d : Storage) (cell : Cell α) : Int :=
  match d with
  | .boolean => if cell.read .int = 0 then 0 else 1
  | .signed _ => cell.read .int
  | .unsigned _ => (cell.read .nat : Int)

/-- The unspecified masked value remains in the original storage channel. -/
def loadValue {α : Type} [Inhabited α] (d : Storage) (s : State α)
    (region : RegionName) (o : Nat) (active : Bool) : Int :=
  match d with
  | .boolean =>
      let v := if active then (s.mem region o).read .int else s.undef .int region o
      if v = 0 then 0 else 1
  | .signed _ => if active then (s.mem region o).read .int else s.undef .int region o
  | .unsigned _ =>
      ((if active then (s.mem region o).read .nat else s.undef .nat region o) : Int)

theorem eval_loadOp {α : Type} [Inhabited α] (M : Algebra α)
    (d : Storage) (region : RegionName) (off : Op .nat shape)
    (mask : Op .bool shape) (s : State α) :
    FP.Structural.evalOp M none (loadOp d region off mask) s = (do
      let offsets ← FP.Structural.evalOp M none off s
      let active ← FP.Structural.evalOp M none mask s
      pure (fun i => loadValue d s region (offsets i) (active i))) := by
  cases ho : FP.Structural.evalOp M none off s <;>
    cases hm : FP.Structural.evalOp M none mask s <;> cases d <;>
    simp [loadOp, FP.Structural.evalOp, ho, hm, bind, pure,
      numericEq, Region.cast, loadValue, bop,
      IntegerExpr.same_left, IntegerExpr.same_right]

def stored {α : Type} (d : Storage) (v : Int) : Cell α :=
  match d with
  | .boolean => .mk .int (BitVec.ofInt 8 v).toInt
  | .signed width => .mk .int (BitVec.ofInt width v).toInt
  | .unsigned width => .mk .nat (BitVec.ofInt width v).toNat

@[simp] theorem loadValue_setReg {α : Type} [Inhabited α]
    (d : Storage) (s : State α) (name : RegName) (dtype : TileDType)
    (sh : TileShape) (value : Values α dtype sh)
    (region : RegionName) (o : Nat) (active : Bool) :
    loadValue d (s.setReg name dtype sh value) region o active =
      loadValue d s region o active := by cases d <;> rfl

/-- Native promotion-width addition, followed by storage conversion. -/
def resultCell {α : Type} [Inhabited α] (x y : Storage) (p : Parameters)
    (mem : RegionName → Nat → Cell α) (o : Nat) : Cell α :=
  let c := promote x y
  stored x (castValue c
    (castValue c (loaded x (mem p.x o)) + castValue c (loaded y (mem p.y o))))

#axiomsClean eval_castOp
#axiomsClean eval_loadOp
#axiomsClean loadValue_setReg
end
end VeriTile.Bench.TritonBenchG.VectorAddition.IntegerSource
