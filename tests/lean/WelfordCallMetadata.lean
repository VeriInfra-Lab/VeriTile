import VeriTile.Triton.DSL.WelfordCallMetadata
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.WelfordCallMetadata
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- Moving the index cast through a shape view changes neither lookup
failure nor any lane value, for every interpretation of floating operations. -/
theorem index_cast_view (M : Algebra α) (p : Option ComputeDType) (s : State α)
    (B : Nat) (index : TileIndex [1,B] → TileIndex [B]) :
    evalOp M p (.castNatToInt (.remap [1,B] index (.ref .nat [B] "values"))) s =
      evalOp M p (.remap [1,B] index (.castNatToInt (.ref .nat [B] "values"))) s := by
  simp [evalOp_unfold,Option.bind]
  cases h : s.regs .nat [B] "values" <;> rfl

theorem constructor_cast (M : Algebra α) (p : Option ComputeDType) (s : State α) (n : Nat) :
    evalOp M p (.full [] (.castNatToInt (.constNat n))) s =
      evalOp M p (.castNatToInt (.full [] (.constNat n))) s := by
  simp [evalOp_unfold,Option.bind]

/-- The source full constructor's signed scalar is exactly the natural
stride passed to the signed range interpreter, including B = 0. -/
theorem initialized_stride (M : Algebra α) (p : Option ComputeDType) (B : Nat)
    (s : State α) (h : s.regs .int [] "step" = some (fun _ => (B : Int))) :
    evalOp M p (.castIntToNat (.ref .int [] "step")) s = some (fun _ => B) := by
  simp [evalOp_unfold,h]

def signature : CallSignature where
  constexpr := ["B"]

/-- error: Welford constexpr dimensions and propagated steps must remain immutable -/
#guard_msgs in
def rebound (B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  with_triton_call signature call_args [] welford_tree tree in triton {
    step = tl.full((), B, tl.int64)
    step = tl.full((), 1, tl.int64)
  }

/-- error: Welford constexpr dimensions and propagated steps must remain immutable -/
#guard_msgs in
def tupleRebound (B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  with_triton_call signature call_args [] welford_tree tree in triton {
    step = tl.full((), B, tl.int64)
    step, _ = tl.broadcast(tl.cast(3, tl.int64), tl.cast(3, tl.int64))
  }

/-- error: Welford constexpr dimensions and propagated steps must remain immutable -/
#guard_msgs in
def tupleConstexprWrite (B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  with_triton_call signature call_args [] welford_tree tree in triton {
    B, other = tl.max(tl.arange(0, B))
  }

/-- error: a loop cannot shadow a Welford dimension or propagated step -/
#guard_msgs in
def shadowed (B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  with_triton_call signature call_args [] welford_tree tree in triton {
    step = tl.full((), B, tl.int64)
    for step in range(0, B):
        value = step
  }

/-- error: Welford constexpr dimensions and propagated steps must remain immutable -/
#guard_msgs in
def constexprWrite (B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  with_triton_call signature call_args [] welford_tree tree in triton {
    B: tl.constexpr = 2
  }

/-- error: Welford metadata supports the original single-row axis 1 helper -/
#guard_msgs in
def wrongAxis (B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  with_triton_call signature call_args [] welford_tree tree in triton {
    mean = tl.zeros([1, B], tl.float32)
    m2 = tl.zeros([1, B], tl.float32)
    weight = tl.zeros([1, B], tl.float32)
    a, b, c = triton_helpers.welford(mean, m2, weight, 0)
  }

#axiomsClean index_cast_view
#axiomsClean constructor_cast
#axiomsClean initialized_stride
end
end VeriTile.Tests.WelfordCallMetadata
