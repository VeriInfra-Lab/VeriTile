import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FPMinimumReduction
open VeriTile.Triton FP.Structural
noncomputable section

def source (input : RegionName) (N : Nat) : ComputeKernel := triton {
  tl.region input = tl.float32
  x = tl.load(input + tl.arange(0, $(N)))
  lo = tl.min(x, 0)
}

theorem source_shape (input : RegionName) (N : Nat) :
    (source input N).surfaceBody[1]? = some
      (.assign .real [] "lo" (.compute (.alg .fp32
        (.reduceMin ⟨0, by simp⟩ Bool.false (.ref .real [N] "x"))))) := rfl

/-- error: tl.min: only real-channel reductions are supported; typed and integer reductions require dedicated lowering -/
#guard_msgs in
example (input : Region .int) (N : Nat) : ComputeKernel := triton {
  x = tl.load(input + tl.arange(0, $(N)))
  lo = tl.min(x, 0)
}

/-- The selected primitive sees the actual precision, complete input tile,
axis and keep-dimensions flag. No maximum or negation callback is consulted. -/
theorem dispatch {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) (n : Nat) (hn : 0 < n)
    (xs : Values α .real [n])
    (hx : s.regs .real [n] "x" = some xs) :
    evalOp M p (.reduceMin ⟨0, by simp⟩ Bool.false (.ref .real [n] "x")) s =
      M.reduceMin p ⟨0, by simp⟩ Bool.false xs := by
  simp [evalOp_unfold, hx, TileShape.axisDim, hn]

theorem unavailable_fails {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) (n : Nat)
    (xs : Values α .real [n]) (hx : s.regs .real [n] "x" = some xs) :
    evalOp { M with reduceMin := fun _ {_} _ _ _ => none } p
      (.reduceMin ⟨0, by simp⟩ Bool.false (.ref .real [n] "x")) s = none := by
  simp [evalOp_unfold, hx, TileShape.axisDim]

theorem empty_fails {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α)
    (xs : Values α .real [0]) (hx : s.regs .real [0] "x" = some xs) :
    evalOp M p (.reduceMin ⟨0, by simp⟩ Bool.false (.ref .real [0] "x")) s = none := by
  simp [evalOp_unfold, hx, TileShape.axisDim]

theorem default_precision {α : Type} (M : Algebra α) (p : ComputeDType)
    (xs : Values α .real [3]) :
    (M.withDefaultPrecision p).reduceMin none ⟨0, by simp⟩ Bool.false xs =
      M.reduceMin (some p) ⟨0, by simp⟩ Bool.false xs := rfl

#axiomsClean source_shape
#axiomsClean dispatch
#axiomsClean unavailable_fails
#axiomsClean empty_fails
#axiomsClean default_precision
end
end VeriTile.Tests.FPMinimumReduction
