import VeriTile.Triton.DSL.SourceProcedures
import VeriTile.Triton.DSL.RebindingProcedures
import VeriTile.Triton.DSL.SpecializedJitFunctions
import VeriTile.Triton.DSL.ClosedUnaryJitFunctions
import VeriTile.Triton.DSL.UniformJitFunctions
import VeriTile.Triton.DSL.ImportedBinaryJitFunctions
import VeriTile.Triton.DSL.ImportedFiniteJitFunctions
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SourceAuthoring
open VeriTile.Triton
noncomputable section

def unaryLegacy : PureJitFunction := triton_function (x) {
    return x * tl.sigmoid(x)
  }
def unarySource : PureJitFunction :=
  with_triton_params (x) constexpr_args [] in triton {
    return x * tl.sigmoid(x)
  }
theorem unary_exact : unarySource = unaryLegacy := rfl

def binaryLegacy : BinaryPureJitFunction := triton_function (a, b) {
    return a // b * b
  }
def binarySource : BinaryPureJitFunction :=
  with_triton_params (a, b) constexpr_args [] in triton {
    return a // b * b
  }
theorem binary_exact : binarySource = binaryLegacy := rfl

def tupleLegacy : StatementJitFunction :=
  triton_function (x, B) constexpr_args [B] {
    y = x + 1.0
    return x, y
  }
def tupleSource : StatementJitFunction :=
  with_triton_params (x, B) constexpr_args [B] in triton {
    y = x + 1.0
    return x, y
  }
theorem tuple_exact : tupleSource = tupleLegacy := rfl

def closedLegacy : ClosedUnaryJitFunction :=
  triton_function (x) constexpr_args [] {
    return x * factor
  }
def closedSource : ClosedUnaryJitFunction :=
  with_triton_params (x) constexpr_args [] in triton {
    return x * factor
  }
theorem closed_exact : closedSource = closedLegacy := rfl

def profile : CallSignature where
  pointers := [("X", .fp32), ("Y", .fp32)]
  scalars := [("N", .integer)]

def effectLegacy : ProcedureJitFunction :=
  triton_procedure profile (X, Y, N) constexpr_args [] {
    value = tl.load(X + 0)
    tl.store(Y + 0, value)
  }
def effectSource : ProcedureJitFunction :=
  with_triton_params (X, Y, N) constexpr_args [] signature profile in triton {
    value = tl.load(X + 0)
    tl.store(Y + 0, value)
  }
theorem effect_exact : effectSource = effectLegacy := rfl

def reboundLegacy : ProcedureJitFunction :=
  triton_rebinding_procedure profile (X, Y, N) constexpr_args [] {
    N = tl.cast(N, tl.int64)
    value = tl.load(X + 0)
    tl.store(Y + 0, value)
  }
def reboundSource : ProcedureJitFunction :=
  with_triton_params (X, Y, N) constexpr_args [] rebinding_signature profile in triton {
    N = tl.cast(N, tl.int64)
    value = tl.load(X + 0)
    tl.store(Y + 0, value)
  }
theorem rebound_exact : reboundSource = reboundLegacy := rfl

theorem reject_unused_fp32_metadata : True := by
  fail_if_success
    let invalid : StatementJitFunction :=
      with_triton_params (x, B) constexpr_args [B] fp32_constants [B] in triton {
        return x
      }
  trivial

#axiomsClean unary_exact
#axiomsClean binary_exact
#axiomsClean tuple_exact
#axiomsClean closed_exact
#axiomsClean effect_exact
#axiomsClean rebound_exact
#axiomsClean reject_unused_fp32_metadata

end
end VeriTile.Tests.SourceAuthoring
