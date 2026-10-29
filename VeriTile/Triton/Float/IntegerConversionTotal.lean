import VeriTile.Triton.Float.IntegerConversion

/-!
An explicit arbitrary-value extension for masked source programs. The shared
partial converter still fails on NaN/Inf/out-of-range input. A task may bind the
undefined result to an arbitrary function, then prove all observed outputs are
independent of that function. This module never claims that hardware returns
zero (or any other chosen value) outside the defined domain.
-/
namespace VeriTile.Triton.FP.IntegerConversion

def totalSigned {f : Format} (undefinedValue : FP.Value f → Int)
    (width : Nat) (x : FP.Value f) : Int :=
  (toSigned width x).getD (undefinedValue x)

def totalUnsigned {f : Format} (undefinedValue : FP.Value f → Nat)
    (width : Nat) (x : FP.Value f) : Nat :=
  (toUnsigned width x).getD (undefinedValue x)

@[simp] theorem totalSigned_defined {f : Format} (undefinedValue : FP.Value f → Int)
    (width : Nat) (x : FP.Value f) (z : Int) (h : toSigned width x = some z) :
    totalSigned undefinedValue width x = z := by simp [totalSigned, h]

@[simp] theorem totalUnsigned_defined {f : Format} (undefinedValue : FP.Value f → Nat)
    (width : Nat) (x : FP.Value f) (z : Nat) (h : toUnsigned width x = some z) :
    totalUnsigned undefinedValue width x = z := by simp [totalUnsigned, h]

/-- Undefined results remain visibly dependent on the arbitrary policy. -/
@[simp] theorem totalSigned_undefined {f : Format} (undefinedValue : FP.Value f → Int)
    (width : Nat) (x : FP.Value f) (h : toSigned width x = none) :
    totalSigned undefinedValue width x = undefinedValue x := by simp [totalSigned, h]

@[simp] theorem totalUnsigned_undefined {f : Format} (undefinedValue : FP.Value f → Nat)
    (width : Nat) (x : FP.Value f) (h : toUnsigned width x = none) :
    totalUnsigned undefinedValue width x = undefinedValue x := by simp [totalUnsigned, h]

noncomputable def fp32TotalBackend (rest : Structural.Algebra (FP.Value .fp32))
    (undefinedSigned : Nat → FP.Value .fp32 → Int)
    (undefinedUnsigned : Nat → FP.Value .fp32 → Nat) : Structural.Algebra (FP.Value .fp32) :=
  { rest with
    floatToInt := fun p src width => if fp32Source p src then
      some (fun x => some (totalSigned (undefinedSigned width) width x)) else none
    floatToNat := fun p src width => if fp32Source p src then
      some (fun x => some (totalUnsigned (undefinedUnsigned width) width x)) else none }

#axiomsClean totalSigned_defined
#axiomsClean totalUnsigned_defined
#axiomsClean totalSigned_undefined
#axiomsClean totalUnsigned_undefined
end VeriTile.Triton.FP.IntegerConversion
