import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural

namespace VeriTile.Tests.CallScalarPrecision
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("alpha",.fp32)]

def scale (X Y : RegionName) (alpha : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [alpha := alpha] in triton {
    x = tl.load(X)
    y = x * alpha
    tl.store(Y, y)
  }

/-- Substitution retains the FP32 literal conversion selected by the loaded
operand. It does not turn the runtime argument into an untyped real operation. -/
theorem scale_source (X Y : RegionName) (alpha : ℝ) :
    scale X Y alpha = (triton {
      tl.region X = tl.float32, Y = tl.float32
      x = tl.load(X)
      y = x * $(alpha)
      tl.store(Y, y)
    }) := rfl

def scalarOnlySignature : CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("alpha",.fp32)]

def scalarOnly (Y : RegionName) (alpha : ℝ) : ComputeKernel :=
  with_triton_call scalarOnlySignature call_args [alpha := alpha] in triton {
    value = -alpha
    tl.store(Y, value)
  }

/-- A supplied FP32 scalar selects FP32 evaluation before any tensor load.
The caller value cannot become a weak real just because this is scalar code. -/
theorem scalar_operand_precision (Y : RegionName) (alpha : ℝ) :
    (scalarOnly Y alpha).surfaceBody[0]? = some
      (.assign .real [] "value" (.compute (.alg .fp32 (.negFloat .real (.const alpha))))) := rfl

def wideSignature : CallSignature where
  pointers := [("X",.fp64),("Y",.fp64)]
  scalars := [("alpha",.fp32)]

def widening (X Y : RegionName) (alpha : ℝ) : ComputeKernel :=
  with_triton_call wideSignature call_args [alpha := alpha] in triton {
    x = tl.load(X)
    y = x * alpha
    tl.store(Y, y)
  }

/-- FP64 arithmetic keeps the FP32 conversion of its runtime scalar input
inside its own precision scope before the wider operation. -/
theorem widening_input_precision {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (X Y : RegionName) (alpha : ℝ) (s : FP.Structural.State α) :
    ∃ t, FP.Structural.exec M (widening X Y alpha) s = some t ∧
      t.mem Y 0 = FP.Structural.Cell.mk .real
        (M.binary (some .fp64) .real .mul ((s.mem X 0).read .real)
          (M.literal (some .fp32) .real alpha)) := by
  simp [widening, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    ComputeDType.eraseDType, FP.Structural.bop, FP.Structural.numeric,
    FP.Structural.store, FP.Structural.State.setReg, FP.Structural.State.write,
    TileShape.allIndices, Region.cast]

def constexprSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["epsilon"]

def addEpsilon (X Y : RegionName) (epsilon : ℝ) : ComputeKernel :=
  with_triton_signature constexprSignature in triton {
    x = tl.load(X)
    y = x + epsilon
    tl.store(Y, y)
  }

theorem epsilon_source (X Y : RegionName) (epsilon : ℝ) :
    addEpsilon X Y epsilon = (triton {
      tl.region X = tl.float32, Y = tl.float32
      x = tl.load(X)
      y = x + $(epsilon)
      tl.store(Y, y)
    }) := rfl

def countSignature : CallSignature where
  pointers := [("Y",.fp32)]
  constexpr := ["B"]

def countKernel (Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature countSignature in triton {
    count = tl.full((), B, tl.uint64).to(tl.float32)
    tl.store(Y, count)
  }

/-- Unsigned count storage narrows before the opaque FP32 integer conversion.
This holds even outside the representable range; no no-wrap fact is assumed. -/
theorem count_semantics {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (Y : RegionName) (B : Nat) (s : FP.Structural.State α) :
    ∃ t, FP.Structural.exec M (countKernel Y B) s = some t ∧
      t.mem Y 0 = FP.Structural.Cell.mk .real (M.fromNat (some .fp32) (B % 2^64)) := by
  simp [countKernel, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    ComputeDType.eraseDType, FP.Structural.bop, IntegerExpr.sameShape,
    FP.Structural.integralMod, FP.Structural.store, FP.Structural.State.setReg,
    FP.Structural.State.write, TileShape.allIndices, Region.cast]

#axiomsClean widening_input_precision
#axiomsClean count_semantics
#axiomsClean scalar_operand_precision
#axiomsClean scale_source
#axiomsClean epsilon_source
end
end VeriTile.Tests.CallScalarPrecision
