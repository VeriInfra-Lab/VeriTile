import VeriTile.Triton.DSL.Syntax
import Lean

/-! Ordinary Lean Boolean terms alongside the Triton expression grammar.
All source frontends share these declarations so joint imports do not create
ambiguous copies of the same Boolean notation. -/
namespace VeriTile.Triton.DSL
macro "true" : term => `(Bool.true)
macro "false" : term => `(Bool.false)
end VeriTile.Triton.DSL
