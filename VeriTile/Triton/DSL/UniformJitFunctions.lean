import VeriTile.Triton.DSL.PureJitFunctions
import VeriTile.Triton.Random.Philox

/-! The canonical dtype-dispatch uniform helper, supplied as copied source.
The direct Philox4x32 call has only 32-bit results, so its statically selected
helper path is signed bitcast, sign fold, conversion and the source FP32 scale.
The unused 64-bit source branch is retained and checked, not executed or claimed
as an additional calling profile. Arbitrary helper bodies are rejected. -/
namespace VeriTile.Triton

namespace DSL.UniformJitFunctions
def canonicalSource : String :=
  "if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):\n" ++
  "    x = x.to(tl.int32, bitcast=True)\n" ++
  "    scale = 4.6566127342e-10\n" ++
  "else:\n" ++
  "    tl.static_assert(\n" ++
  "        tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64)\n" ++
  "    )\n" ++
  "    x = x.to(tl.int64, bitcast=True)\n" ++
  "    scale = 1.0842020432385337e-19\n" ++
  "x = tl.where(x < 0, -x - 1, x)\n" ++
  "return x * scale"

def normalizedCanonical : String :=
  "if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32) {\nx = x.to(tl.int32, bitcast=true)\nscale = 4.6566127342e-10\n} else {\ntl.static_assert( tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64) )\nx = x.to(tl.int64, bitcast=true)\nscale = 1.0842020432385337e-19\n}\nx = tl.where(x < 0, -x - 1, x)\nreturn x * scale\n"

end DSL.UniformJitFunctions

def UniformJitFunction.Canonical (parameter source : String) : Prop :=
  parameter="x" ∧ source.toList.filter (!·.isWhitespace) ==
    DSL.UniformJitFunctions.normalizedCanonical.toList.filter (!·.isWhitespace)

structure UniformJitFunction where
  parameter : String
  source : String
  canonical : UniformJitFunction.Canonical parameter source

namespace UniformJitFunction
/-- The actual selected 32-bit helper operations in the fixed software model. -/
def eval32 (_helper : UniformJitFunction) (word : Random.Word) : FP.Value .fp32 :=
  let x := word.toInt
  let folded := if x < 0 then -x-1 else x
  let scale := FP.round .fp32 FP.Config.ieeeValue (46566127342 / 10^20 : Rat)
  FP.mul FP.Config.ieeeValue (FP.round .fp32 FP.Config.ieeeValue (Rat.ofInt folded)) scale
end UniformJitFunction

namespace DSL.UniformJitFunctions
open Lean Elab Term Meta

/-- Restrict the helper to the complete pinned source; whitespace alone may
vary. No branch, assertion, cast, operation, scale or return may be omitted. -/
def validate (parameter source : String) : TermElabM Unit := do
  let .ok canonical := RawSourceText.normalize ("{\n"++canonicalSource++"\n}")
    | throwError "invalid canonical uniform source"
  unless parameter=="x" && source.toList.filter (!·.isWhitespace) ==
      canonical.toList.filter (!·.isWhitespace) do
    throwError "uniform helper requires the complete canonical dtype-dispatch source"

@[term_elab PureJitFunctions.pureFunction] def elaborateFunction : TermElab := fun stx expected => do
  let some expected := expected | throwUnsupportedSyntax
  unless ← isDefEq expected (mkConst ``UniformJitFunction) do throwUnsupportedSyntax
  let parameter:=stx[2].getId.eraseMacroScopes.toString
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok source => pure source | .error message => throwError "{message}"
  validate parameter source
  let canonical ← mkAppM ``UniformJitFunction.Canonical #[mkStrLit parameter,mkStrLit source]
  let proof ← elabTermEnsuringType (← `(by unfold UniformJitFunction.Canonical; decide +kernel)) canonical
  return ← mkAppM ``UniformJitFunction.mk #[mkStrLit parameter,mkStrLit source,proof]

end DSL.UniformJitFunctions
end VeriTile.Triton
