import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedReverseBody
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 500

def signature : CallSignature where
  pointers := [("Y", .fp32)]
  constexpr := ["B"]

def source (Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    for j in range(tl.cast(B - 1, tl.int64), tl.cast(-1, tl.int64), -1):
        tl.store(Y, j.to(tl.float32))
  }

/-- The subtraction precedes the signed cast: B=0 starts at -1. -/
theorem loop_bounds (Y : RegionName) (B : Nat) :
    ∃ body, (source Y B).surfaceBody =
      [.forRangeInt "__rev_j"
        (.sub .int .nil (.constInt 0) (.constInt (Int.ofNat B - 1)))
        (.sub .int .nil (.constInt 0) (.sub .int .nil (.constInt 0) (.constInt 1)))
        (.constNat 1) body] := by
  exact ⟨_, rfl⟩

/-- An empty Python reverse range performs no writes and binds no iterator. -/
theorem zero_iterations {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (Y : RegionName) (u : FP.Structural.State α) :
    FP.Structural.exec M (source Y 0) u = some u := by
  obtain ⟨body, equality⟩ := loop_bounds Y 0
  change FP.Structural.run M (source Y 0).surfaceBody u = some u
  rw [equality, FP.Structural.run, FP.Structural.step]
  simp only [FP.Structural.evalOp]
  change (FP.Structural.rangeInt M "__rev_j" 1 1 1 body u).bind
    (FP.Structural.run M []) = some u
  rw [FP.Structural.rangeInt]
  simp only [show ¬(1 : Nat) = 0 by decide, show ¬(1 : Int) < 1 by decide,
    if_false, Option.bind_some, FP.Structural.run]

/-- User source may use the old mirrored-index spelling as an ordinary value. -/
def collision (Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    __rev_j = tl.full([], 7, tl.int64)
    for j in range(tl.cast(B - 1, tl.int64), tl.cast(-1, tl.int64), -1):
        tl.store(Y, (__rev_j + j).to(tl.float32))
  }

theorem fresh_iterator (Y : RegionName) (B : Nat) :
    ((collision Y B).surfaceBody.getLast?.map fun statement =>
      match statement with
      | .forRangeInt name _ _ _ _ => name
      | _ => "") = some "__rev_j_" := rfl

#axiomsClean loop_bounds
#axiomsClean zero_iterations
#axiomsClean fresh_iterator
end
end VeriTile.Tests.SignedReverseBody
