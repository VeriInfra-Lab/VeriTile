import bench.tritonbench_g.vector_addition.Proofs.Mathematical_IntegerGrid
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_IntegerProfiles
import VeriTile.Triton.Float.IntegerConversionTotal

/-! The 18 ordered canonical integer/bool and fp32 profiles. Integer-to-fp32
is one concrete rounding step. The fp32 sum is the concrete software FP.add.
Integer outputs use toward-zero conversion at the destination width; bool
storage is signed8, not a nonzero predicate. Undefined conversions have an
explicit arbitrary policy, including on masked lanes. Native correspondence,
tag/byte representability and sanitizer behavior remain separate obligations.
-/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32
open VeriTile.Triton FP.Structural FP.IntegerConversion
open ValueProofs (Parameters offset)
open IntegerSource (Storage Canonical)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- A .real carrier is used solely inside an explicit fp32 precision scope;
the final tag conversion preserves that already-rounded binary32 value. -/
def backend (rest : Algebra Word) (cfg : FP.Config)
    (undefinedSigned : Nat → Word → Int) (undefinedUnsigned : Nat → Word → Nat) :
    Algebra Word :=
  fp32TotalBackend
    { FP.Software.scalarFP32 rest cfg with
      fromInt := fun p n => if p = some .fp32 then FP.round .fp32 cfg (n : Rat)
        else rest.fromInt p n
      fromNat := fun p n => if p = some .fp32 then FP.round .fp32 cfg (n : Rat)
        else rest.fromNat p n
      cast := fun p src dst v =>
        if p = some .fp32 ∧ src = .real ∧ dst = .fp32 then v
        else (FP.Software.scalarFP32 rest cfg).cast p src dst v }
    undefinedSigned undefinedUnsigned

theorem backend_add (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) (a b : Word) :
    (backend rest cfg us uu).binary none .fp32 .add a b = FP.add cfg a b := rfl

theorem backend_signed (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) (width : Nat) :
    (backend rest cfg us uu).floatToInt none .fp32 width =
      some (fun v => some (totalSigned (us width) width v)) := rfl

theorem backend_unsigned (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat) (width : Nat) :
    (backend rest cfg us uu).floatToNat none .fp32 width =
      some (fun v => some (totalUnsigned (uu width) width v)) := rfl

def integerToFloat (d : Storage) (e : Op .int shape) : Op .fp32 shape :=
  .withPrecision .fp32 .fp32 (.castFloat .real .fp32
    (if d.compute.signed then
      .intToReal (IntegerSource.castOp d.compute e)
    else .natToReal (.intToUnsigned d.compute.width e)))

def integerRounded (cfg : FP.Config) (d : Storage) (v : Int) : Word :=
  if d.compute.signed then
    FP.round .fp32 cfg ((IntegerSource.castValue d.compute v : Int) : Rat)
  else FP.round .fp32 cfg (((BitVec.ofInt d.compute.width v).toNat : Nat) : Rat)

/-- Fixed-width cast evaluation also holds under an explicit floating context.
The context is passed through to the argument; no floating operation is added. -/
theorem eval_castOp_at {α : Type} [Inhabited α] (M : Algebra α)
    (precision : Option ComputeDType) (f : IntegerSource.Format)
    (e : Op .int shape) (s : State α) :
    FP.Structural.evalOp M precision (IntegerSource.castOp f e) s = (do
      let v ← FP.Structural.evalOp M precision e s
      pure (fun i => IntegerSource.castValue f (v i))) := by
  unfold IntegerSource.castOp IntegerSource.castValue
  split
  · rw [FP.Structural.eval_unsignedToInt, FP.Structural.eval_intToUnsigned]
    cases h : FP.Structural.evalOp M precision e s <;>
      simp only [h, bind, pure, Option.bind_none, Option.bind_some]
    congr 1
    funext i
    rw [signedBits_eq_bitvec, unsigned_cast_eq_bitvec, BitVec.ofNat_toNat, BitVec.setWidth_eq]
  · simp only [FP.Structural.evalOp, FP.Structural.eval_intToUnsigned]
    cases h : FP.Structural.evalOp M precision e s <;>
      simp only [h, bind, pure, Option.bind_none, Option.bind_some]
    congr 1
    funext i
    simp only [bop, integralMod, IntegerExpr.same_left, IntegerExpr.same_right]
    rw [unsigned_cast_eq_bitvec]
    rfl

theorem eval_natToReal_at {α : Type} [Inhabited α] (M : Algebra α)
    (precision : Option ComputeDType) (e : Op .nat shape) (s : State α) :
    FP.Structural.evalOp M precision (.natToReal e) s = (do
      let v ← FP.Structural.evalOp M precision e s
      pure (fun i => M.fromNat precision (v i))) := rfl

theorem eval_integerToFloat (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (d : Storage) (e : Op .int shape) (s : State Word) :
    FP.Structural.evalOp (backend rest cfg us uu) none (integerToFloat d e) s = (do
      let v ← FP.Structural.evalOp (backend rest cfg us uu) (some .fp32) e s
      pure (fun i => integerRounded cfg d (v i))) := by
  unfold integerToFloat
  by_cases hs : d.compute.signed = Bool.true
  · simp only [hs, ↓reduceIte, FP.Structural.evalOp, eval_castOp_at]
    cases he : FP.Structural.evalOp (backend rest cfg us uu) (some .fp32) e s <;>
      simp [he, backend, fp32TotalBackend, integerRounded, hs, bind, pure,
        ofFloat, toFloat]
  · simp only [hs, ↓reduceIte, FP.Structural.evalOp, eval_natToReal_at,
      FP.Structural.eval_intToUnsigned]
    cases he : FP.Structural.evalOp (backend rest cfg us uu) (some .fp32) e s <;>
      simp only [backend, fp32TotalBackend] at he <;>
      simp [he, backend, fp32TotalBackend, integerRounded, hs, bind, pure,
        FP.Structural.evalOp, eval_natToReal_at, FP.Structural.eval_intToUnsigned,
        bop, integralMod, IntegerExpr.same_left, IntegerExpr.same_right,
        unsigned_cast_eq_bitvec, -BitVec.toNat_ofInt, -BitVec.toInt_ofInt,
        ofFloat, toFloat]

def convertedCell (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (d : Storage) (v : Word) : Cell Word :=
  match d with
  | .boolean => .mk .int (totalSigned (us 8) 8 v)
  | .signed width => .mk .int (totalSigned (us width) width v)
  | .unsigned width => .mk .nat (totalUnsigned (uu width) width v)

def sumValue (cfg : FP.Config) (integerFirst : Bool) (d : Storage)
    (p : Parameters) (mem : RegionName → Nat → Cell Word) (o : Nat) : Word :=
  if integerFirst then FP.add cfg
    (integerRounded cfg d (IntegerSource.loaded d (mem p.x o))) ((mem p.y o).read .fp32)
  else FP.add cfg ((mem p.x o).read .fp32)
    (integerRounded cfg d (IntegerSource.loaded d (mem p.y o)))

def resultCell (cfg : FP.Config) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters)
    (mem : RegionName → Nat → Cell Word) (o : Nat) : Cell Word :=
  let sum := sumValue cfg integerFirst d p mem o
  if integerFirst then convertedCell us uu d sum else .mk .fp32 sum

#axiomsClean backend_add
#axiomsClean backend_signed
#axiomsClean backend_unsigned
#axiomsClean eval_castOp_at
#axiomsClean eval_natToReal_at
#axiomsClean eval_integerToFloat
end
end VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32
