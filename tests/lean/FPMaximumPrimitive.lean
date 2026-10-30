import VeriTile.Triton.DSL
import VeriTile.Triton.Float.MaximumPrimitive
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FPMaximumPrimitive
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def typedSource (dtype : FloatDType) (a b out : RegionName) : ComputeKernel := triton {
  tl.region a = DTYPE, b = DTYPE, out = DTYPE
  x = tl.load(a)
  y = tl.load(b)
  tl.store(out, tl.maximum(x, y))
}

theorem typed_source_shape (dtype : FloatDType) (a b out : RegionName) :
    (typedSource dtype a b out).surfaceBody[2]? = some
      (.store dtype.toTileDType [] (.region out.cast (.constNat 0))
        (.alg (.maximumFloat dtype .nil (.ref dtype.toTileDType [] "x")
          (.ref dtype.toTileDType [] "y"))) .none) := rfl

def compute32Source (a b : RegionName) : ComputeKernel := triton {
  tl.region a = tl.float32, b = tl.float32
  x = tl.load(a)
  y = tl.load(b)
  z = tl.maximum(x, y)
}

theorem compute32_source_shape (a b : RegionName) :
    (compute32Source a b).surfaceBody[2]? = some
      (.assign .real [] "z" (.compute (.alg .fp32
        (.maximumFloat .real .nil (.ref .real [] "x") (.ref .real [] "y"))))) := rfl

def compute64Source (a b : RegionName) : ComputeKernel := triton {
  tl.region a = tl.float64, b = tl.float64
  x = tl.load(a)
  y = tl.load(b)
  z = tl.maximum(x, y)
}

theorem compute64_source_shape (a b : RegionName) :
    (compute64Source a b).surfaceBody[2]? = some
      (.assign .real [] "z" (.compute (.alg .fp64
        (.maximumFloat .real .nil (.ref .real [] "x") (.ref .real [] "y"))))) := rfl

def nestedSource (dtype : FloatDType) (a b : RegionName) (N : Nat) : ComputeKernel := triton {
  tl.region a = DTYPE, b = DTYPE
  x = tl.load(a + tl.arange(0, $(N)))
  y = tl.load(b)
  z = tl.maximum(tl.maximum(x, y), x)
}

theorem nested_broadcast_source_shape (dtype : FloatDType) (a b : RegionName) (N : Nat) :
    (nestedSource dtype a b N).surfaceBody[2]? = some
      (.assign dtype.toTileDType [N] "z" (.alg
        (.maximumFloat dtype (.consSame .nil)
          (.maximumFloat dtype .scalarR (.ref dtype.toTileDType [N] "x")
            (.ref dtype.toTileDType [] "y")) (.ref dtype.toTileDType [N] "x")))) := rfl

def nestedPrecision (a b : RegionName) : ComputeKernel := triton {
  tl.region a = tl.float32, b = tl.float32
  x = tl.load(a)
  y = tl.load(b)
  z = tl.maximum(x, tl.maximum(x, y).to(tl.float64))
}

private def maxSites : {d : TileDType} → {shape : TileShape} →
    Option IndexComputePrecision → Op d shape → List (Option IndexComputePrecision × FloatDType)
  | _, _, p, .maximumFloat dtype _ a b => (p, dtype) :: (maxSites p a ++ maxSites p b)
  | _, _, _, .withPrecision _ p a => maxSites (some p) a
  | _, _, p, .castFloat _ _ a => maxSites p a
  | _, _, _, _ => []

private def assignedSites : ComputeStmt → List (Option IndexComputePrecision × FloatDType)
  | .assign _ _ _ (.alg e) => maxSites none e
  | .assign _ _ _ (.compute (.alg .fp32 e)) => maxSites (some .fp32) e
  | .assign _ _ _ (.compute (.alg .fp64 e)) => maxSites (some .fp64) e
  | _ => []

/-- Widening an inner fp32 maximum must not change that operation's context. -/
theorem nested_precision_preserved (a b : RegionName) :
    ((nestedPrecision a b).surfaceBody[2]?).map assignedSites =
      some [(some .fp64, .real), (some .fp32, .real)] := rfl

def natSource (a b : Region .nat) : ComputeKernel := triton {
  x = tl.load(a)
  y = tl.load(b)
  z = tl.maximum(x, y)
}

theorem integer_maximum_unchanged (a b : Region .nat) :
    (natSource a b).surfaceBody[2]? = some
      (.assign .nat [] "z" (.alg (.where
        (.gt .nat .nil (.ref .nat [] "x") (.ref .nat [] "y"))
        (.ref .nat [] "x") (.ref .nat [] "y")))) := rfl

theorem typed_source_execution {α : Type} [Inhabited α] (M : Algebra α)
    (dtype : FloatDType) (a b out : RegionName) (s : State α) :
    ∃ final, exec M (typedSource dtype a b out) s = some final ∧
      final.mem = (s.write out 0 (.mk dtype.toTileDType
        (ofFloat dtype (M.binary none dtype .max
          (toFloat dtype ((s.mem a 0).read dtype.toTileDType))
          (toFloat dtype ((s.mem b 0).read dtype.toTileDType)))))).mem := by
  cases dtype <;> simp [typedSource, exec, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, store, bop,
    TileShape.allIndices, Broadcast.leftIndex, Broadcast.rightIndex,
    State.setReg, State.write, toFloat, ofFloat, Region.cast, Option.bind]

abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
def one : Word := FP.Value.ofBits .fp32 0x3f800000
def nan : Word := FP.Value.ofBits .fp32 0x7fc12345

/-- A concrete test profile with max-number handling of a single NaN. Its
both-NaN canonicalization and left zero tie policy are software choices;
this regression does not assert a complete native PTX conformance result. -/
def maxNumber (x y : Word) : Word :=
  if x.isNaN then (if y.isNaN then FP.Value.qNaN .fp32 else y)
  else if y.isNaN then x
  else if FP.lt .ieeeValue x y then y else x

def maxBackend (rest : Algebra Word) : Algebra Word :=
  { rest with binary := fun p dtype op x y =>
      if p = none ∧ dtype = .fp32 ∧ op = .max then maxNumber x y
      else rest.binary p dtype op x y }

def initial (flip : Bool) : State Word where
  mem := fun r _ => .mk .fp32
    (if r = ("a" : RegionName) then (if flip then nan else one)
     else if flip then one else nan)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem single_nan_values : maxNumber one nan = one ∧ maxNumber nan one = one ∧
    (if FP.lt .ieeeValue nan one then one else nan) = nan ∧ one ≠ nan := by decide +kernel

theorem source_handles_both_nan_positions (rest : Algebra Word) (flip : Bool) :
    ∃ final, exec (maxBackend rest) (typedSource .fp32 "a" "b" "out") (initial flip) = some final ∧
      final.mem "out" 0 = .mk .fp32 one ∧
      final.mem "a" 0 = (initial flip).mem "a" 0 := by
  obtain ⟨final, he, hm⟩ := typed_source_execution (maxBackend rest) .fp32 "a" "b" "out" (initial flip)
  refine ⟨final, he, ?_, ?_⟩
  all_goals rw [hm]
  all_goals cases flip <;> simp [State.write, initial, Cell.read, maxBackend,
    toFloat, ofFloat, single_nan_values.1, single_nan_values.2.1]

#axiomsClean typed_source_shape
#axiomsClean compute32_source_shape
#axiomsClean compute64_source_shape
#axiomsClean nested_broadcast_source_shape
#axiomsClean nested_precision_preserved
#axiomsClean integer_maximum_unchanged
#axiomsClean typed_source_execution
#axiomsClean single_nan_values
#axiomsClean source_handles_both_nan_positions
end
end VeriTile.Tests.FPMaximumPrimitive
